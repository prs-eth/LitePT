"""Per-class NMS-radius calibration for a trained landmark model.

decode_predictions runs greedy radius-NMS independently per class (see
models/landmark.py:greedy_radius_nms), and the challenge metric scores each
class independently too (metrics/landmark.py:eval_det_cls_map) -- so the
optimal nms_radius for e.g. Cusp does not depend on what radius is used for
Mesial, and classes can be swept one at a time instead of as a combinatorial
grid. tools/tune_landmarks.py already covers the (score_threshold, nms_radius,
max_predictions) grid with a single shared radius; this is the per-class
counterpart for nms_radius alone, using the same "one forward pass, cache raw
heat_logits/proposals" trick so every radius is a free re-filter of the cache.

score_threshold is intentionally not swept here: at the low threshold used in
training/eval the challenge metric is dominated by high-score duplicates, not
by how deep the output is flooded, so radius is what needs tuning.

Usage:
    python tools/tune_nms_per_class.py \
        --config-file exp/landmark/preliminary_grid_search/configs/base_lc4_np16384.py \
        --weight exp/landmark/preliminary_grid_search/base_lc4_np16384/model/model_best.pth \
        --split val [--fold <split_folder>] [--gold gold_standard.pkl] \
        [--radii 0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0] \
        [--objective mean|ap|ar] [--output-csv per_class_nms.csv]

Prints, per class, the AP/AR at every candidate radius and the best pick, then
the full calibrated list in class order ready to paste into a config's
`model = dict(nms_radius=[...])`.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from datasets import build_dataset, collate_fn
from eval_landmarks import load_weight
from metrics.landmark import eval_det_cls_map, gold_from_dataset, load_gold, subset_gold, voc_ar
from models import build_model
from models.landmark import greedy_radius_nms
from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--weight", required=True, help="path to model checkpoint")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument(
        "--fold",
        default=None,
        help="split folder with train/validation/test lists; defaults to the config's fold",
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="challenge gold_standard.pkl; if omitted, GT is read from the __kpt.json files",
    )
    parser.add_argument(
        "--radii", nargs="+", type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    )
    parser.add_argument(
        "--objective", choices=("mean", "ap", "ar"), default="mean",
        help="metric to maximize per class when picking the best radius",
    )
    parser.add_argument(
        "--output-csv", default=None, help="defaults to <save_path>/nms_per_class_<split>.csv"
    )
    return parser.parse_args()


def class_ap_ar(gt_cls, pred_map_cls):
    """metrics.landmark.score(), specialized to a single class in isolation."""
    ap_values, recall_values, exp_neg_thresh = [], [], []
    for i in range(30):
        dist_thresh = 0.1 * i
        rec, _, ap = eval_det_cls_map(pred_map_cls, gt_cls, dist_thresh)
        ap_values.append(ap)
        recall_values.append(rec[-1] if len(rec) else 0.0)
        exp_neg_thresh.append(np.exp(-dist_thresh))
    mean_ap = float(np.mean(ap_values))
    mean_ar = float(voc_ar(exp_neg_thresh, recall_values))
    return mean_ap, mean_ar


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    class_names = cfg.data.names

    data_cfg = cfg.data[args.split]
    if args.fold is not None:
        data_cfg.fold = args.fold
    dataset = build_dataset(data_cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(cfg.model).cuda()
    load_weight(model, args.weight)
    model.eval()

    if args.gold is not None:
        keys = {Path(p).stem for p in dataset.data_list}
        gold = subset_gold(load_gold(args.gold), keys)
    else:
        gold = gold_from_dataset(dataset, class_names)

    # single forward pass; cache the raw per-point scores/proposals, sorted
    # by nothing yet -- greedy_radius_nms sorts internally per call below
    scans = []
    for input_dict in loader:
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.no_grad():
            output_dict = model(input_dict)
        scans.append(
            {
                "name": input_dict["name"][0],
                "probs": torch.sigmoid(output_dict["heat_logits"]).cpu(),
                "proposals": output_dict["proposals"].cpu(),
            }
        )
        print(f"[{len(scans)}/{len(loader)}] {scans[-1]['name']}")

    # score_threshold matches the model's real decode floor: without it every
    # raw point (up to num_points per scan) enters the O(N*K) greedy loop below
    # instead of the handful that would ever reach NMS in practice
    score_threshold = model.score_threshold
    print(f"score_threshold={score_threshold} (from model config)")

    results = []
    calibrated = {}
    for cls_idx, cls_name in enumerate(class_names):
        if cls_name not in gold:
            continue
        print(f"class {cls_idx + 1}/{len(class_names)}: {cls_name}")
        per_radius = []
        for r_idx, radius in enumerate(args.radii):
            print(f"  [{r_idx + 1}/{len(args.radii)}] radius={radius:.2f} ...", end="", flush=True)
            pred_map = {}
            num_candidates = 0
            for scan in scans:
                coords, scores = greedy_radius_nms(
                    scan["proposals"][:, cls_idx, :],
                    scan["probs"][:, cls_idx],
                    radius,
                    score_threshold,
                )
                pred_map[scan["name"]] = list(zip(coords.tolist(), scores.tolist()))
                num_candidates += len(coords)
            ap, ar = class_ap_ar(gold[cls_name], pred_map)
            objective = {"ap": ap, "ar": ar, "mean": (ap + ar) / 2}[args.objective]
            per_radius.append(
                {"class": cls_name, "nms_radius": radius, "AP": ap, "AR": ar, "objective": objective}
            )
            print(f" {num_candidates} kept -> AP {ap:.4f}  AR {ar:.4f}")

        best = max(per_radius, key=lambda r: r["objective"])
        calibrated[cls_name] = best
        results.extend(per_radius)
        print(f"  -> best {cls_name}: radius={best['nms_radius']} (AP {best['AP']:.4f}, AR {best['AR']:.4f})")

    output_csv = args.output_csv or os.path.join(cfg.save_path, f"nms_per_class_{args.split}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "nms_radius", "AP", "AR", "objective"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {output_csv}")

    radii_in_order = [calibrated[name]["nms_radius"] for name in class_names if name in calibrated]
    mean_ap = np.mean([calibrated[name]["AP"] for name in class_names if name in calibrated])
    mean_ar = np.mean([calibrated[name]["AR"] for name in class_names if name in calibrated])
    print(f"\nCombined at calibrated radii: mAP {mean_ap:.4f}  mAR {mean_ar:.4f}")
    print(f"class order: {list(class_names)}")
    print(f"nms_radius = {radii_in_order}")


if __name__ == "__main__":
    main()
