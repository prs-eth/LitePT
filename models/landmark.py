import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.structure import Point
from .builder import MODELS, build_model


DEFAULT_LANDMARK_CLASSES = (
    "Mesial",
    "Distal",
    "InnerPoint",
    "OuterPoint",
    "FacialPoint",
    "Cusp",
)


def greedy_radius_nms(coords, scores, radius, score_threshold=0.0, max_predictions=None):
    """Greedy NMS for one class: score-descending, keep points >= radius apart.

    coords: (N, 3), scores: (N,), any device. Returns (kept_coords, kept_scores)
    on cpu, sorted by descending score. Shared by decode_predictions (per class,
    per scan) and offline post-hoc calibration over cached raw predictions.
    """
    order = torch.argsort(scores, descending=True)
    scores = scores[order].detach().cpu()
    coords = coords[order].detach().cpu()

    kept_coords, kept_scores = [], []
    for coord, score in zip(coords, scores):
        if score.item() < score_threshold:
            break
        too_close = any(
            torch.norm(coord - kept_coord).item() < radius for kept_coord in kept_coords
        )
        if too_close:
            continue
        kept_coords.append(coord)
        kept_scores.append(score)
        if max_predictions is not None and len(kept_coords) >= max_predictions:
            break

    if kept_coords:
        return torch.stack(kept_coords, dim=0), torch.stack(kept_scores, dim=0)
    return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)


@MODELS.register_module()
class LandmarkDetector(nn.Module):
    def __init__(
        self,
        num_classes=6,
        backbone_out_channels=72,
        backbone=None,
        class_names=DEFAULT_LANDMARK_CLASSES,
        positive_radius=2.0,
        smooth_l1_beta=1.0,
        lambda_coord=1.0,
        lambda_focal=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        heat_target_gaussian=False,
        gaussian_sigma_ratio=1.0 / 3.0,
        score_threshold=0.5,
        nms_radius=2.0,
        max_predictions_per_class=None,
        predict_tooth=False,
        num_tooth_classes=17,
        lambda_tooth=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = tuple(class_names)
        # positive_radius, smooth_l1_beta and nms_radius are all expressed in
        # original coordinate units (mm); coord_scale converts per sample.
        self.positive_radius = positive_radius
        self.smooth_l1_beta = smooth_l1_beta
        self.lambda_coord = lambda_coord
        self.lambda_focal = lambda_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # heat target within positive_radius: hard 1.0 (default) or a Gaussian
        # falloff by distance to the matched landmark, sigma = positive_radius
        # * gaussian_sigma_ratio (see loss()). positive_radius itself still
        # defines the pos/neg split and what counts for coord regression --
        # only the heat/focal target value changes.
        self.heat_target_gaussian = heat_target_gaussian
        self.gaussian_sigma_ratio = gaussian_sigma_ratio
        self.score_threshold = score_threshold
        # auxiliary supervision: classify each predicted landmark's tooth index
        # (nearest-vertex label from the Teeth3DS segmentation mask). Disabled
        # by default; requires the dataset to be built with load_segment=True
        # so "landmark_tooth" is present in input_dict.
        self.predict_tooth = predict_tooth
        self.num_tooth_classes = num_tooth_classes
        self.lambda_tooth = lambda_tooth
        # scalar applies the same radius to every class; a sequence of
        # num_classes values allows per-class calibration (see
        # tools/tune_nms_per_class.py)
        if isinstance(nms_radius, (list, tuple)):
            assert len(nms_radius) == num_classes, "nms_radius sequence must match num_classes"
            self.nms_radius = tuple(nms_radius)
        else:
            self.nms_radius = (nms_radius,) * num_classes
        self.max_predictions_per_class = max_predictions_per_class

        self.backbone = build_model(backbone)
        self.heat_head = nn.Linear(backbone_out_channels, num_classes)
        self.offset_head = nn.Linear(backbone_out_channels, num_classes * 3)
        self.tooth_head = (
            nn.Linear(backbone_out_channels, num_classes * num_tooth_classes)
            if predict_tooth
            else None
        )

    def _extract_feat(self, input_dict):
        point = self.backbone(Point(input_dict))
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return point.feat
        return point

    def forward(self, input_dict):
        feat = self._extract_feat(input_dict)
        heat_logits = self.heat_head(feat)
        offsets = self.offset_head(feat).view(-1, self.num_classes, 3)
        tooth_logits = None
        if self.predict_tooth:
            tooth_logits = self.tooth_head(feat).view(-1, self.num_classes, self.num_tooth_classes)
        normalized_proposals = input_dict["coord"].unsqueeze(1) + offsets

        if "landmark_coord" in input_dict and "landmark_class" in input_dict:
            loss, loss_dict = self.loss(input_dict, heat_logits, normalized_proposals, tooth_logits)
            if self.training:
                return {"loss": loss, **loss_dict}
        else:
            loss, loss_dict = None, None

        proposals = normalized_proposals
        output = {
            "heat_logits": heat_logits,
            "offsets": offsets,
        }
        if self.predict_tooth:
            output["tooth_logits"] = tooth_logits
        if not self.training:
            proposals = self._proposals_to_original_space(input_dict, normalized_proposals)
            output["offsets_original"] = self._offsets_to_original_space(input_dict, offsets)

        output["proposals"] = proposals

        if loss is not None:
            output["loss"] = loss
            output["loss_dict"] = loss_dict

        if not self.training:
            output["pred_landmarks"] = self.decode_predictions(
                input_dict, heat_logits, proposals
            )
        return output

    def _offsets_to_original_space(self, input_dict, offsets):
        point_offset = input_dict["offset"].detach().cpu().tolist()
        point_start = [0] + point_offset[:-1]
        scales = input_dict["coord_scale"]
        offsets_original = offsets.clone()
        for ps, pe, scale in zip(point_start, point_offset, scales):
            offsets_original[ps:pe] = offsets[ps:pe] * scale.to(
                device=offsets.device, dtype=offsets.dtype
            )
        return offsets_original

    def _proposals_to_original_space(self, input_dict, proposals):
        point_offset = input_dict["offset"].detach().cpu().tolist()
        point_start = [0] + point_offset[:-1]
        centers = input_dict["coord_center"]
        scales = input_dict["coord_scale"]
        proposals_original = proposals.clone()
        for ps, pe, center, scale in zip(point_start, point_offset, centers, scales):
            center = center.to(device=proposals.device, dtype=proposals.dtype)
            scale = scale.to(device=proposals.device, dtype=proposals.dtype)
            proposals_original[ps:pe] = proposals[ps:pe] * scale + center
        return proposals_original

    def _sigmoid_focal_loss(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(self.focal_gamma) * bce).sum()

    def _sample_ranges(self, input_dict):
        point_offset = input_dict["offset"].detach().cpu().tolist()
        point_start = [0] + point_offset[:-1]

        landmark_offset = input_dict["landmark_offset"].detach().cpu().tolist()
        landmark_start = [0] + landmark_offset[:-1]
        return zip(point_start, point_offset, landmark_start, landmark_offset)

    def loss(self, input_dict, heat_logits, proposals, tooth_logits=None):
        coord_loss = proposals.sum() * 0.0
        focal_loss = heat_logits.sum() * 0.0
        num_positives = 0

        landmark_coord = input_dict["landmark_coord"]
        landmark_class = input_dict["landmark_class"]
        coord = input_dict["coord"]
        scales = input_dict["coord_scale"].to(device=coord.device, dtype=torch.float32)

        # tooth labels may be absent for a given batch even when predict_tooth
        # is enabled -- e.g. a split whose segmentation masks aren't available
        # (see Teeth3DLandmarkDataset._report_segment_availability). Skip the
        # term for that batch rather than failing.
        has_tooth_labels = self.predict_tooth and "landmark_tooth" in input_dict
        if has_tooth_labels:
            landmark_tooth = input_dict["landmark_tooth"]
            tooth_loss = tooth_logits.sum() * 0.0

        for sample_idx, (ps, pe, ls, le) in enumerate(self._sample_ranges(input_dict)):
            scale = scales[sample_idx].clamp_min(1e-6)
            radius = self.positive_radius / scale
            sample_classes = landmark_class[ls:le]
            sample_gt = landmark_coord[ls:le]
            sample_coord = coord[ps:pe]
            if has_tooth_labels:
                sample_tooth = landmark_tooth[ls:le]

            for cls_idx in range(self.num_classes):
                logits = heat_logits[ps:pe, cls_idx]
                props = proposals[ps:pe, cls_idx, :]
                gt = sample_gt[sample_classes == cls_idx]
                targets = torch.zeros_like(logits)

                if gt.numel() > 0 and props.shape[0] > 0:
                    # assign each point to its nearest same-class landmark;
                    # points within positive_radius become positives and
                    # regress their proposal onto that landmark
                    dist = torch.cdist(sample_coord.detach().float(), gt.detach().float(), p=2)
                    nearest_dist, nearest_idx = dist.min(dim=1)
                    pos_mask = nearest_dist <= radius

                    if pos_mask.any():
                        if self.heat_target_gaussian:
                            sigma = (radius * self.gaussian_sigma_ratio).clamp_min(1e-6)
                            gaussian_target = torch.exp(
                                -0.5 * (nearest_dist[pos_mask] / sigma) ** 2
                            )
                            targets[pos_mask] = gaussian_target.to(targets.dtype)
                        else:
                            targets[pos_mask] = 1.0
                        # residual in original units (mm) so beta and the loss
                        # magnitude are independent of per-scan normalization
                        residual = (props[pos_mask] - gt[nearest_idx[pos_mask]]) * scale
                        coord_loss = coord_loss + F.smooth_l1_loss(
                            residual,
                            torch.zeros_like(residual),
                            reduction="sum",
                            beta=self.smooth_l1_beta,
                        )
                        num_positives += int(pos_mask.sum().item())

                        if has_tooth_labels:
                            # ground-truth tooth is the mask label of each
                            # landmark's nearest mesh vertex, precomputed in
                            # the dataset; positives regress onto their
                            # assigned landmark's tooth, like the coord head
                            gt_tooth = sample_tooth[sample_classes == cls_idx]
                            tooth_targets = gt_tooth[nearest_idx[pos_mask]]
                            tooth_pred = tooth_logits[ps:pe, cls_idx, :][pos_mask]
                            tooth_loss = tooth_loss + F.cross_entropy(
                                tooth_pred, tooth_targets, reduction="sum"
                            )

                focal_loss = focal_loss + self._sigmoid_focal_loss(logits, targets)

        norm = max(num_positives, 1)
        coord_loss = coord_loss / norm
        focal_loss = focal_loss / norm

        total_loss = (
            self.lambda_coord * coord_loss
            + self.lambda_focal * focal_loss
        )
        loss_dict = {
            "coord_loss": coord_loss.detach(),
            "focal_loss": focal_loss.detach(),
            "num_positives": torch.as_tensor(num_positives, device=heat_logits.device),
        }

        if has_tooth_labels:
            tooth_loss = tooth_loss / norm
            total_loss = total_loss + self.lambda_tooth * tooth_loss
            loss_dict["tooth_loss"] = tooth_loss.detach()

        return total_loss, loss_dict

    def decode_predictions(self, input_dict, heat_logits, proposals):
        probs = torch.sigmoid(heat_logits)
        point_offset = input_dict["offset"].detach().cpu().tolist()
        point_start = [0] + point_offset[:-1]
        batch_predictions = []

        for ps, pe in zip(point_start, point_offset):
            pred_coords = []
            pred_classes = []
            pred_scores = []

            for cls_idx in range(self.num_classes):
                kept_coords, kept_scores = greedy_radius_nms(
                    proposals[ps:pe, cls_idx, :],
                    probs[ps:pe, cls_idx],
                    self.nms_radius[cls_idx],
                    self.score_threshold,
                    self.max_predictions_per_class,
                )
                pred_coords.extend(kept_coords)
                pred_classes.extend([cls_idx] * len(kept_coords))
                pred_scores.extend(kept_scores)

            if pred_coords:
                batch_predictions.append(
                    {
                        "coord": torch.stack(pred_coords, dim=0),
                        "class": torch.as_tensor(pred_classes, dtype=torch.long),
                        "score": torch.stack(pred_scores, dim=0),
                    }
                )
            else:
                batch_predictions.append(
                    {
                        "coord": torch.zeros((0, 3), dtype=torch.float32),
                        "class": torch.zeros((0,), dtype=torch.long),
                        "score": torch.zeros((0,), dtype=torch.float32),
                    }
                )
        return batch_predictions
