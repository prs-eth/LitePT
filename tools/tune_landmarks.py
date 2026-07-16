"""Grid search over the post-processing parameters of a trained landmark model.

Sweeps the inference-time parameters that need no retraining — score_threshold,
nms_radius, max_predictions_per_class — and scores every combination with the
official challenge metric. The model forward runs only once per scan: each
combination just re-filters the cached predictions. Results are written to a
csv (one row per combination, sorted by mAP) ready for Excel.

Usage:
    python tools/tune_landmarks.py \
        --config-file configs/teeth3ds/landmark-litept-small.py \
        --weight exp/landmark/<exp>/model/model_best.pth \
        --split val [--fold <split_folder>] [--gold gold_standard.pkl] \
        [--score-thresholds 0.05 0.1 0.3] [--nms-radii 1.0 2.0] \
        [--max-predictions 16 32] [--output-csv grid_search.csv]
"""

import argparse
import csv
import itertools
import os
from pathlib import Path

import torch

from datasets import build_dataset, collate_fn
from eval_landmarks import load_weight
from metrics.landmark import (
    challenge_score,
    gold_from_dataset,
    load_gold,
    prediction_rows,
    subset_gold,
)
from models import build_model
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
        "--score-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument(
        "--nms-radii", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0]
    )
    parser.add_argument("--max-predictions", nargs="+", type=int, default=[16, 32])
    parser.add_argument(
        "--output-csv", default=None, help="defaults to <save_path>/grid_search_<split>.csv"
    )
    return parser.parse_args()


def filter_prediction(pred, score_threshold, max_per_class, num_classes):
    """Restrict a decoded prediction to a stricter threshold and per-class cap.

    decode_predictions returns landmarks grouped by class and sorted by
    descending score, and its greedy NMS decisions only depend on higher-scored
    candidates — so the result for a stricter threshold/cap is exactly the
    loose result filtered by score and truncated per class.
    """
    keep = []
    for cls_idx in range(num_classes):
        cls_indices = (pred["class"] == cls_idx).nonzero(as_tuple=True)[0]
        cls_indices = cls_indices[pred["score"][cls_indices] >= score_threshold]
        keep.append(cls_indices[:max_per_class])
    keep = torch.cat(keep)
    return {key: pred[key][keep] for key in ("coord", "class", "score")}


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

    # single forward pass; cache heatmaps and original-space proposals per scan
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
                "offset": input_dict["offset"].cpu(),
                "heat_logits": output_dict["heat_logits"].cpu(),
                "proposals": output_dict["proposals"].cpu(),
            }
        )
        print(f"[{len(scans)}/{len(loader)}] {scans[-1]['name']}")

    combos = list(
        itertools.product(args.nms_radii, args.score_thresholds, args.max_predictions)
    )
    base_threshold = min(args.score_thresholds)
    results = []
    for nms_radius in args.nms_radii:
        # decode once per radius at the loosest threshold without a cap; every
        # (threshold, cap) combination is a plain filter of this result
        model.nms_radius = nms_radius
        model.score_threshold = base_threshold
        model.max_predictions_per_class = None
        decoded = [
            model.decode_predictions(
                {"offset": scan["offset"]}, scan["heat_logits"], scan["proposals"]
            )[0]
            for scan in scans
        ]
        for score_threshold, max_pred in itertools.product(
            args.score_thresholds, args.max_predictions
        ):
            rows = []
            for scan, pred in zip(scans, decoded):
                filtered = filter_prediction(
                    pred, score_threshold, max_pred, len(class_names)
                )
                rows.extend(prediction_rows(scan["name"], filtered, class_names))
            metrics = challenge_score(rows, gold)
            results.append(
                {
                    "nms_radius": nms_radius,
                    "score_threshold": score_threshold,
                    "max_predictions_per_class": max_pred,
                    "num_predictions": len(rows),
                    **metrics,
                }
            )
            print(
                "[{}/{}] nms_radius={} score_threshold={} max_predictions={} -> "
                "mAP {:.4f}, mAR {:.4f}".format(
                    len(results), len(combos), nms_radius, score_threshold, max_pred,
                    metrics["mAP"], metrics["mAR"],
                )
            )

    results.sort(key=lambda r: r["mAP"], reverse=True)
    output_csv = args.output_csv or os.path.join(
        cfg.save_path, f"grid_search_{args.split}.csv"
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} combinations to {output_csv}")

    best = results[0]
    print(
        "Best: nms_radius={nms_radius} score_threshold={score_threshold} "
        "max_predictions={max_predictions_per_class} -> mAP {mAP:.4f}, mAR {mAR:.4f}".format(**best)
    )


if __name__ == "__main__":
    main()
