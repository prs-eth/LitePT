import os
import json
import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
import plotly.graph_objects as go
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from uuid import uuid4

import utils.comm as comm
from utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS

from datasets.transform import Compose
from metrics.landmark import (
    challenge_score,
    gold_from_dataset,
    load_gold,
    prediction_rows,
    project_to_surface,
    subset_gold,
)
from metrics.semantic import ConfusionMatrix

from torch_scatter import scatter_mean
from torch.nn.functional import normalize
from tqdm import tqdm


def _mesh_vertices_for_batch(dataset, names, full_paths, cache):
    """Look up each scan's full-resolution mesh vertices, caching per name.

    Used to snap predictions back onto the mesh surface -- the model only
    ever sees the FPS/random-subsampled points, so this reloads the full
    mesh the dataset already read once (and discarded) while building the
    batch.
    """
    vertices = []
    for name, full_path in zip(names, full_paths):
        if name not in cache:
            cache[name] = dataset._load_mesh(full_path)[0]
        vertices.append(cache[name])
    return vertices


def _project_predictions_to_surface(pred_landmarks, vertices_list):
    projected = []
    for pred, vertices in zip(pred_landmarks, vertices_list):
        coord = pred["coord"]
        if coord.shape[0] == 0:
            projected.append(pred)
            continue
        snapped = project_to_surface(coord.detach().cpu().numpy(), vertices)
        projected.append(
            {**pred, "coord": torch.from_numpy(snapped).to(device=coord.device, dtype=coord.dtype)}
        )
    return projected


@HOOKS.register_module()
class LandmarkEvaluator(HookBase):
    def __init__(
        self,
        output_dir_name="validation_output",
        max_visualization_scans=8,
        max_visualization_points=8192,
        max_monitor_values=200000,
        debug_top_candidates_per_class=1,
        match_distance_threshold=3.0,
        project_to_surface=False,
    ):
        self.output_dir_name = output_dir_name
        self.max_visualization_scans = max_visualization_scans
        self.max_visualization_points = max_visualization_points
        self.max_monitor_values = max_monitor_values
        self.debug_top_candidates_per_class = debug_top_candidates_per_class
        # maximum prediction-GT distance (original units, mm) for a Hungarian
        # pair to count as a match in precision/recall/mean_error
        self.match_distance_threshold = match_distance_threshold
        # snap predictions to the nearest mesh vertex before scoring/visualizing
        self.project_to_surface = project_to_surface
        self._mesh_cache = {}

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Landmark Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        loss_sum = torch.zeros(1, device="cuda")
        batch_count = torch.zeros(1, device="cuda")
        error_sum = torch.zeros(1, device="cuda")
        match_count = torch.zeros(1, device="cuda")
        gt_count = torch.zeros(1, device="cuda")
        pred_count = torch.zeros(1, device="cuda")
        validation_records = []
        visualization_records = []
        regression_monitor = {}
        classification_monitor = {}

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            self._update_regression_monitor(regression_monitor, output_dict)
            self._update_classification_monitor(classification_monitor, input_dict, output_dict)

            loss = output_dict["loss"]
            loss_sum += loss.detach()
            batch_count += 1

            pred_landmarks = output_dict["pred_landmarks"]
            if self.project_to_surface:
                vertices_list = _mesh_vertices_for_batch(
                    self.trainer.val_loader.dataset,
                    input_dict["name"],
                    input_dict["full_path"],
                    self._mesh_cache,
                )
                pred_landmarks = _project_predictions_to_surface(pred_landmarks, vertices_list)

            metric = self._compute_batch_metric(input_dict, pred_landmarks)
            error_sum += metric["error_sum"]
            match_count += metric["match_count"]
            gt_count += metric["gt_count"]
            pred_count += metric["pred_count"]
            remaining_visualizations = None
            if self.max_visualization_scans is not None:
                remaining_visualizations = max(
                    self.max_visualization_scans - len(visualization_records), 0
                )
            batch_records, batch_visualizations = self._build_prediction_records(
                input_dict,
                pred_landmarks,
                remaining_visualizations,
                output_dict,
            )
            validation_records.extend(batch_records)
            visualization_records.extend(batch_visualizations)

            self.trainer.logger.info(
                "Val: [{iter}/{max_iter}] Loss {loss:.4f}".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        stats = torch.cat(
            [loss_sum, batch_count, error_sum, match_count, gt_count, pred_count]
        )
        if comm.get_world_size() > 1:
            dist.all_reduce(stats)
            gathered_records = [None for _ in range(comm.get_world_size())]
            gathered_visualizations = [None for _ in range(comm.get_world_size())]
            gathered_regression_monitors = [None for _ in range(comm.get_world_size())]
            gathered_classification_monitors = [None for _ in range(comm.get_world_size())]
            dist.all_gather_object(gathered_records, validation_records)
            dist.all_gather_object(gathered_visualizations, visualization_records)
            dist.all_gather_object(gathered_regression_monitors, regression_monitor)
            dist.all_gather_object(gathered_classification_monitors, classification_monitor)
            validation_records = [item for part in gathered_records for item in part]
            visualization_records = [item for part in gathered_visualizations for item in part]
            regression_monitor = self._merge_regression_monitors(gathered_regression_monitors)
            classification_monitor = self._merge_regression_monitors(gathered_classification_monitors)

        loss_avg = (stats[0] / stats[1].clamp_min(1)).item()
        mean_error = (stats[2] / stats[3].clamp_min(1)).item()
        recall = (stats[3] / stats[4].clamp_min(1)).item()
        precision = (stats[3] / stats[5].clamp_min(1)).item()
        current_metric = -mean_error if stats[3].item() > 0 else -loss_avg

        self.trainer.comm_info["current_metric_value"] = current_metric
        self.trainer.comm_info["current_metric_name"] = "neg_mean_landmark_error"
        self.trainer.logger.info(
            "Val result: loss {:.4f}, mean_error {:.4f}, precision {:.4f}, recall {:.4f}.".format(
                loss_avg, mean_error, precision, recall
            )
        )

        current_epoch = self.trainer.epoch + 1
        regression_stats, regression_samples = self._finalize_regression_monitor(regression_monitor)
        classification_stats, classification_samples = self._finalize_regression_monitor(
            classification_monitor
        )
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mean_error", mean_error, current_epoch)
            self.trainer.writer.add_scalar("val/precision", precision, current_epoch)
            self.trainer.writer.add_scalar("val/recall", recall, current_epoch)
            for key, value in regression_stats.items():
                self.trainer.writer.add_scalar(f"val/regression/{key}", value, current_epoch)
            for name, samples in regression_samples.items():
                self.trainer.writer.add_histogram(f"val/regression/{name}", samples, current_epoch)
            for key, value in classification_stats.items():
                self.trainer.writer.add_scalar(f"val/classification/{key}", value, current_epoch)
            for name, samples in classification_samples.items():
                self.trainer.writer.add_histogram(
                    f"val/classification/{name}", samples, current_epoch
                )
            if self.trainer.cfg.enable_wandb:
                wandb_payload = {
                    f"val/regression/{key}": value
                    for key, value in regression_stats.items()
                }
                wandb_payload.update(
                    {
                        f"val/classification/{key}": value
                        for key, value in classification_stats.items()
                    }
                )
                for name, samples in regression_samples.items():
                    wandb_payload[f"val/regression/{name}_hist"] = wandb.Histogram(
                        samples.cpu().numpy()
                    )
                for name, samples in classification_samples.items():
                    wandb_payload[f"val/classification/{name}_hist"] = wandb.Histogram(
                        samples.cpu().numpy()
                    )
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mean_error": mean_error,
                        "val/precision": precision,
                        "val/recall": recall,
                        **wandb_payload,
                    },
                    step=wandb.run.step,
                )

        for name in regression_samples.keys():
            prefix = f"{name}/"
            self.trainer.logger.info(
                "Val regression {name}: mean {mean:.4f}, std {std:.4f}, abs_mean {abs_mean:.4f}, "
                "min {min:.4f}, max {max:.4f}, p95_abs {p95_abs:.4f}".format(
                    name=name,
                    mean=regression_stats[f"{prefix}mean"],
                    std=regression_stats[f"{prefix}std"],
                    abs_mean=regression_stats[f"{prefix}abs_mean"],
                    min=regression_stats[f"{prefix}min"],
                    max=regression_stats[f"{prefix}max"],
                    p95_abs=regression_stats[f"{prefix}p95_abs"],
                )
            )
        for name in (
            "heat_scores",
            "heat_top_scores_per_class",
            "heat_scores_above_threshold",
            "heat_top_scores_above_threshold",
            "decoded_landmark_count",
        ):
            if name not in classification_samples:
                continue
            prefix = f"{name}/"
            self.trainer.logger.info(
                "Val classification {name}: mean {mean:.4f}, min {min:.4f}, max {max:.4f}, "
                "p95 {p95:.4f}".format(
                    name=name,
                    mean=classification_stats[f"{prefix}mean"],
                    min=classification_stats[f"{prefix}min"],
                    max=classification_stats[f"{prefix}max"],
                    p95=classification_stats[f"{prefix}p95"],
                )
            )

        if comm.is_main_process():
            self._save_validation_outputs(current_epoch, validation_records, visualization_records)

    def _update_regression_monitor(self, monitor, output_dict):
        for name in ("offsets", "offsets_original", "proposals"):
            if name not in output_dict:
                continue
            values = output_dict[name].detach().float().reshape(-1)
            values = values[torch.isfinite(values)]
            self._update_tensor_monitor(monitor, name, values)

    def _update_classification_monitor(self, monitor, input_dict, output_dict):
        if "heat_logits" not in output_dict:
            return

        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        score_threshold = getattr(model, "score_threshold", 0.5)
        heat_logits = output_dict["heat_logits"].detach().float()
        heat_scores = torch.sigmoid(heat_logits)
        self._update_tensor_monitor(monitor, "heat_logits", heat_logits.reshape(-1))
        self._update_tensor_monitor(monitor, "heat_scores", heat_scores.reshape(-1))
        self._update_tensor_monitor(
            monitor,
            "heat_scores_above_threshold",
            (heat_scores.reshape(-1) >= score_threshold).float(),
        )

        point_offset = input_dict["offset"].detach().cpu().tolist()
        point_start = [0] + point_offset[:-1]
        top_scores = []
        decoded_counts = []
        for pred, ps, pe in zip(output_dict["pred_landmarks"], point_start, point_offset):
            decoded_counts.append(float(pred["score"].numel()))
            if pe <= ps:
                continue
            top_scores.append(heat_scores[ps:pe].max(dim=0).values)

        if top_scores:
            top_scores = torch.cat(top_scores)
            self._update_tensor_monitor(monitor, "heat_top_scores_per_class", top_scores)
            self._update_tensor_monitor(
                monitor,
                "heat_top_scores_above_threshold",
                (top_scores >= score_threshold).float(),
            )
        if decoded_counts:
            self._update_tensor_monitor(
                monitor,
                "decoded_landmark_count",
                torch.as_tensor(decoded_counts, device=heat_logits.device),
            )

    def _update_tensor_monitor(self, monitor, name, values):
        values = values.detach().float().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return

        stat = monitor.setdefault(
            name,
            {
                "count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "abs_sum": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "samples": [],
                "sample_count": 0,
            },
        )
        stat["count"] += int(values.numel())
        stat["sum"] += float(values.sum().item())
        stat["sumsq"] += float(values.square().sum().item())
        stat["abs_sum"] += float(values.abs().sum().item())
        stat["min"] = min(stat["min"], float(values.min().item()))
        stat["max"] = max(stat["max"], float(values.max().item()))

        remaining = self.max_monitor_values - stat["sample_count"]
        if remaining <= 0:
            return
        take = min(remaining, values.numel())
        if take == values.numel():
            samples = values.detach().cpu()
        else:
            index = torch.linspace(0, values.numel() - 1, take, device=values.device).long()
            samples = values[index].detach().cpu()
        stat["samples"].append(samples)
        stat["sample_count"] += int(samples.numel())

    def _merge_regression_monitors(self, monitors):
        merged = {}
        for monitor in monitors:
            for name, stat in monitor.items():
                target = merged.setdefault(
                    name,
                    {
                        "count": 0,
                        "sum": 0.0,
                        "sumsq": 0.0,
                        "abs_sum": 0.0,
                        "min": float("inf"),
                        "max": float("-inf"),
                        "samples": [],
                        "sample_count": 0,
                    },
                )
                target["count"] += stat["count"]
                target["sum"] += stat["sum"]
                target["sumsq"] += stat["sumsq"]
                target["abs_sum"] += stat["abs_sum"]
                target["min"] = min(target["min"], stat["min"])
                target["max"] = max(target["max"], stat["max"])

                remaining = self.max_monitor_values - target["sample_count"]
                if remaining <= 0:
                    continue
                samples = torch.cat(stat["samples"]) if stat["samples"] else torch.empty(0)
                if samples.numel() > remaining:
                    index = torch.linspace(0, samples.numel() - 1, remaining).long()
                    samples = samples[index]
                target["samples"].append(samples)
                target["sample_count"] += int(samples.numel())
        return merged

    def _finalize_regression_monitor(self, monitor):
        stats = {}
        samples_by_name = {}
        for name, stat in monitor.items():
            count = max(stat["count"], 1)
            mean = stat["sum"] / count
            variance = max(stat["sumsq"] / count - mean * mean, 0.0)
            samples = torch.cat(stat["samples"]) if stat["samples"] else torch.empty(0)
            if samples.numel() == 0:
                continue

            abs_samples = samples.abs()
            quantiles = torch.quantile(samples, torch.tensor([0.05, 0.5, 0.95]))
            abs_quantiles = torch.quantile(abs_samples, torch.tensor([0.95, 0.99]))
            prefix = f"{name}/"
            stats.update(
                {
                    f"{prefix}mean": mean,
                    f"{prefix}std": float(np.sqrt(variance)),
                    f"{prefix}abs_mean": stat["abs_sum"] / count,
                    f"{prefix}min": stat["min"],
                    f"{prefix}max": stat["max"],
                    f"{prefix}p05": float(quantiles[0].item()),
                    f"{prefix}p50": float(quantiles[1].item()),
                    f"{prefix}p95": float(quantiles[2].item()),
                    f"{prefix}p95_abs": float(abs_quantiles[0].item()),
                    f"{prefix}p99_abs": float(abs_quantiles[1].item()),
                }
            )
            samples_by_name[name] = samples
        return stats, samples_by_name

    def _compute_batch_metric(self, input_dict, predictions):
        landmark_coord = input_dict["landmark_coord"].detach().cpu()
        landmark_class = input_dict["landmark_class"].detach().cpu()
        landmark_offset = input_dict["landmark_offset"].detach().cpu().tolist()
        landmark_start = [0] + landmark_offset[:-1]
        centers = input_dict["coord_center"].detach().cpu()
        scales = input_dict["coord_scale"].detach().cpu()

        error_sum = 0.0
        match_count = 0
        gt_count = 0
        pred_count = 0
        num_classes = self.trainer.cfg.data.num_classes

        for pred, ls, le, center, scale in zip(
            predictions, landmark_start, landmark_offset, centers, scales
        ):
            gt_coord = landmark_coord[ls:le] * scale.item() + center
            gt_class = landmark_class[ls:le]
            pred_coord = pred["coord"]
            pred_class = pred["class"]

            for cls_idx in range(num_classes):
                gt = gt_coord[gt_class == cls_idx]
                pr = pred_coord[pred_class == cls_idx]
                gt_count += int(gt.shape[0])
                pred_count += int(pr.shape[0])
                if gt.numel() == 0 or pr.numel() == 0:
                    continue
                dist_mat = torch.cdist(pr.float(), gt.float(), p=2).numpy()
                row_idx, col_idx = linear_sum_assignment(dist_mat)
                match_dist = dist_mat[row_idx, col_idx]
                if self.match_distance_threshold is not None:
                    match_dist = match_dist[match_dist <= self.match_distance_threshold]
                error_sum += float(match_dist.sum())
                match_count += len(match_dist)

        device = input_dict["coord"].device
        return {
            "error_sum": torch.tensor([error_sum], device=device),
            "match_count": torch.tensor([match_count], device=device, dtype=torch.float32),
            "gt_count": torch.tensor([gt_count], device=device, dtype=torch.float32),
            "pred_count": torch.tensor([pred_count], device=device, dtype=torch.float32),
        }

    def _build_prediction_records(
        self, input_dict, predictions, remaining_visualizations, output_dict=None
    ):
        names = input_dict["name"]
        full_paths = input_dict.get("full_path", [None] * len(names))
        centers = input_dict["coord_center"].detach().cpu()
        scales = input_dict["coord_scale"].detach().cpu()
        coords = input_dict["coord"].detach().cpu()
        point_offset = input_dict["offset"].detach().cpu().tolist()
        point_start = [0] + point_offset[:-1]
        class_names = self.trainer.cfg.data.names
        heat_scores = None
        proposals = None
        if output_dict is not None and "heat_logits" in output_dict and "proposals" in output_dict:
            heat_scores = torch.sigmoid(output_dict["heat_logits"].detach().cpu())
            proposals = output_dict["proposals"].detach().cpu()

        records = []
        visualizations = []
        for sample_idx, (name, full_path, pred, center, scale, ps, pe) in enumerate(
            zip(names, full_paths, predictions, centers, scales, point_start, point_offset)
        ):
            pred_coord = pred["coord"]
            pred_class = pred["class"]
            pred_score = pred["score"]
            objects = []
            for obj_idx, (coord, cls_idx, score) in enumerate(
                zip(pred_coord, pred_class, pred_score)
            ):
                cls_idx = int(cls_idx.item())
                objects.append(
                    {
                        "key": f"{name}_pred_{obj_idx}",
                        "class": class_names[cls_idx],
                        "class_index": cls_idx,
                        "coord": [float(v) for v in coord.tolist()],
                        "score": float(score.item()),
                    }
                )

            debug_objects = []
            if heat_scores is not None and proposals is not None:
                for cls_idx, cls_name in enumerate(class_names):
                    cls_scores = heat_scores[ps:pe, cls_idx]
                    if cls_scores.numel() == 0:
                        continue
                    topk = min(self.debug_top_candidates_per_class, cls_scores.numel())
                    top_scores, top_indices = torch.topk(cls_scores, k=topk)
                    cls_props = proposals[ps:pe, cls_idx]
                    for rank, (score, index) in enumerate(zip(top_scores, top_indices)):
                        coord = cls_props[index]
                        debug_objects.append(
                            {
                                "class": cls_name,
                                "class_index": cls_idx,
                                "coord": [float(v) for v in coord.tolist()],
                                "score": float(score.item()),
                                "rank": rank + 1,
                            }
                        )

            records.append(
                {
                    "name": name,
                    "full_path": full_path,
                    "objects": objects,
                }
            )

            if remaining_visualizations is None or len(visualizations) < remaining_visualizations:
                scan_coord = coords[ps:pe] * scale.item() + center
                if (
                    self.max_visualization_points is not None
                    and scan_coord.shape[0] > self.max_visualization_points
                ):
                    index = torch.linspace(
                        0,
                        scan_coord.shape[0] - 1,
                        self.max_visualization_points,
                    ).long()
                    scan_coord = scan_coord[index]
                visualizations.append(
                    {
                        "name": name,
                        "points": scan_coord.numpy().tolist(),
                        "objects": objects,
                        "debug_objects": debug_objects,
                    }
                )

        return records, visualizations

    def _save_validation_outputs(self, current_epoch, records, visualizations):
        output_dir = os.path.join(self.trainer.cfg.save_path, self.output_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, f"epoch_{current_epoch:04d}_predictions.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "epoch": current_epoch,
                    "classes": list(self.trainer.cfg.data.names),
                    "scans": records,
                },
                f,
                indent=2,
            )

        html_path = os.path.join(output_dir, f"epoch_{current_epoch:04d}_visualization.html")
        with open(html_path, "w") as f:
            f.write(self._render_validation_html(current_epoch, visualizations))

        self.trainer.logger.info(f"Saved validation predictions to: {json_path}")
        self.trainer.logger.info(f"Saved validation visualization to: {html_path}")

        if self.trainer.cfg.enable_wandb and wandb.run is not None:
            wandb.log(
                {"val/landmark_visualization": wandb.Html(html_path)},
                step=wandb.run.step,
            )

    def _render_validation_html(self, current_epoch, visualizations):
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
        class_names = list(self.trainer.cfg.data.names)
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8' /></head><body>",
            f"<h2>Validation Landmark Predictions - Epoch {current_epoch}</h2>",
        ]

        for scan_idx, scan in enumerate(visualizations):
            fig = go.Figure()
            points = np.asarray(scan["points"], dtype=np.float32)
            if points.size > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode="markers",
                        name="scan points",
                        marker=dict(size=1.5, color="#999999", opacity=0.25),
                    )
                )

            for cls_idx, cls_name in enumerate(class_names):
                debug_objects = [
                    obj
                    for obj in scan.get("debug_objects", [])
                    if obj["class_index"] == cls_idx
                ]
                if debug_objects:
                    debug_coords = np.asarray(
                        [obj["coord"] for obj in debug_objects], dtype=np.float32
                    )
                    debug_scores = [f"top {obj['score']:.2f}" for obj in debug_objects]
                    fig.add_trace(
                        go.Scatter3d(
                            x=debug_coords[:, 0],
                            y=debug_coords[:, 1],
                            z=debug_coords[:, 2],
                            mode="markers+text",
                            name=f"{cls_name} top candidate",
                            text=debug_scores,
                            textposition="bottom center",
                            marker=dict(
                                size=4,
                                color=colors[cls_idx % len(colors)],
                                symbol="x",
                                opacity=0.55,
                            ),
                        )
                    )

                objects = [obj for obj in scan["objects"] if obj["class_index"] == cls_idx]
                if not objects:
                    continue
                coords = np.asarray([obj["coord"] for obj in objects], dtype=np.float32)
                scores = [f"{obj['score']:.2f}" for obj in objects]
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode="markers+text",
                        name=cls_name,
                        text=scores,
                        textposition="top center",
                        marker=dict(
                            size=6,
                            color=colors[cls_idx % len(colors)],
                            opacity=0.95,
                        ),
                    )
                )

            fig.update_layout(
                title=scan["name"],
                scene=dict(aspectmode="data"),
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h"),
                height=650,
            )
            html_parts.append(
                fig.to_html(
                    full_html=False,
                    include_plotlyjs="cdn" if scan_idx == 0 else False,
                )
            )

        html_parts.append("</body></html>")
        return "\n".join(html_parts)


@HOOKS.register_module()
class LandmarkChallengeScorer(HookBase):
    """Score splits with the official 3DTeethLand challenge metric after each eval epoch.

    Runs inference on each split and computes the mAP/mAR reported by the
    challenge scoring script (see metrics/landmark.py). Ground truth comes from
    a gold_standard.pkl if given, otherwise from the splits' __kpt.json files.
    """

    def __init__(self, splits=("val", "test"), gold_path=None, project_to_surface=False):
        self.splits = splits
        self.gold_path = gold_path
        # snap predictions to the nearest mesh vertex before scoring
        self.project_to_surface = project_to_surface
        self.loaders = {}
        self.gold = {}
        self._mesh_cache = {}

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            for split in self.splits:
                wandb.define_metric(f"{split}/*", step_metric="Epoch")

        class_names = self.trainer.cfg.data.names
        full_gold = load_gold(self.gold_path) if self.gold_path else None
        for split in self.splits:
            loader = self._build_loader(split)
            self.loaders[split] = loader
            if full_gold is None:
                self.gold[split] = gold_from_dataset(loader.dataset, class_names)
                continue
            names = {Path(p).stem for p in loader.dataset.data_list}
            self.gold[split] = subset_gold(full_gold, names)
            covered = set().union(*(scans.keys() for scans in self.gold[split].values()))
            if names - covered:
                self.trainer.logger.warning(
                    f"{len(names - covered)} {split} scans are missing from {self.gold_path}; "
                    "their predictions will count as false positives."
                )

    def _build_loader(self, split):
        # deferred import: datasets -> models -> engines.hooks would be circular at module level
        from datasets import build_dataset, collate_fn

        dataset = build_dataset(self.trainer.cfg.data[split])
        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            if comm.get_world_size() > 1
            else None
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.trainer.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Challenge Scoring >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        class_names = self.trainer.cfg.data.names
        current_epoch = self.trainer.epoch + 1

        for split in self.splits:
            rows_by_scan = {}
            mesh_cache = self._mesh_cache.setdefault(split, {})
            for input_dict in self.loaders[split]:
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict)
                pred_landmarks = output_dict["pred_landmarks"]
                if self.project_to_surface:
                    vertices_list = _mesh_vertices_for_batch(
                        self.loaders[split].dataset,
                        input_dict["name"],
                        input_dict["full_path"],
                        mesh_cache,
                    )
                    pred_landmarks = _project_predictions_to_surface(pred_landmarks, vertices_list)
                for name, pred in zip(input_dict["name"], pred_landmarks):
                    rows_by_scan[name] = prediction_rows(name, pred, class_names)

            if comm.get_world_size() > 1:
                gathered = comm.gather(rows_by_scan, dst=0)
                # merging by scan name dedupes samples repeated by DistributedSampler padding
                rows_by_scan = {k: v for part in gathered for k, v in part.items()}
            if not comm.is_main_process():
                continue

            rows = [row for scan_rows in rows_by_scan.values() for row in scan_rows]
            scores = challenge_score(rows, self.gold[split])
            # persist for tools/grid_search.py --collect; on resume epochs may
            # repeat, readers should keep the last record per (split, epoch)
            with open(Path(self.trainer.cfg.save_path) / "challenge_scores.jsonl", "a") as f:
                f.write(json.dumps({"epoch": current_epoch, "split": split, **scores}) + "\n")
            self.trainer.logger.info(
                "Challenge {} result: mAP {:.4f}, mAR {:.4f}.".format(
                    split, scores["mAP"], scores["mAR"]
                )
            )
            if self.trainer.writer is not None:
                # error_mean_mm/error_std_mm can be None (no matched pairs
                # for a class yet, e.g. early epochs) -- skip those rather
                # than feeding None into add_scalar/wandb
                loggable = {k: v for k, v in scores.items() if v is not None}
                for key, value in loggable.items():
                    self.trainer.writer.add_scalar(
                        f"{split}/challenge/{key}", value, current_epoch
                    )
                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            **{f"{split}/challenge/{key}": value for key, value in loggable.items()},
                        },
                        step=wandb.run.step,
                    )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=True):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
        
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )

@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/mAP": all_ap,
                        "val/AP50": all_ap_50,
                        "val/AP25": all_ap_25,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
