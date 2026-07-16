"""Run landmark inference on a split and write a challenge-format predictions.csv.

The csv (key, coord_x, coord_y, coord_z, class, score) can be fed directly to the
challenge scoring script. If a gold_standard.pkl is given (or GT __kpt.json files
are available next to the scans) the challenge mAP/mAR is also computed here.

Usage:
    python tools/eval_landmarks.py \
        --config-file configs/teeth3ds/landmark-litept-small.py \
        --weight exp/landmark/<exp>/model/model_best.pth \
        --split test \
        [--fold /path/to/split_folder] [--gold /path/to/gold_standard.pkl] \
        [--output-dir /path/to/output]
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from datasets import build_dataset, collate_fn
from metrics.landmark import (
    challenge_score,
    gold_from_dataset,
    load_gold,
    prediction_rows,
    subset_gold,
    write_predictions_csv,
)
from models import build_model
from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--weight", required=True, help="path to model checkpoint")
    parser.add_argument("--split", default="test", choices=("val", "test"))
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
        "--output-dir", default=None, help="defaults to <save_path>/eval_<split>"
    )
    return parser.parse_args()


def load_weight(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    weight = OrderedDict(
        (key[len("module."):] if key.startswith("module.") else key, value)
        for key, value in checkpoint["state_dict"].items()
    )
    model.load_state_dict(weight, strict=True)
    print(f"Loaded weight '{weight_path}' (epoch {checkpoint.get('epoch')})")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    class_names = cfg.data.names

    data_cfg = cfg.data[args.split]
    if args.fold is not None:
        data_cfg.fold = args.fold
    dataset = build_dataset(data_cfg)

    model = build_model(cfg.model).cuda()
    load_weight(model, args.weight)
    model.eval()

    # scans are processed one at a time, in-process, so that file I/O can be
    # timed separately from processing (same boundary as the challenge
    # submission scripts, which time everything after the mesh is loaded)
    rows = []
    num_scans = len(dataset.data_list)
    io_time = proc_time = forward_time = 0.0
    start = time.perf_counter()
    for idx, rel_path in enumerate(dataset.data_list):
        scan_path = os.path.join(dataset.data_root, rel_path)

        load_start = time.perf_counter()
        coord = dataset._load_mesh(scan_path)
        landmark_coord, landmark_class = dataset._load_landmarks(scan_path)
        io_time += time.perf_counter() - load_start

        # mirrors Teeth3DLandmarkDataset.get_data from here on
        proc_start = time.perf_counter()
        coord = coord[dataset._fps_indices(coord)]
        data_dict = dataset.transform(
            {
                "coord": coord.astype(np.float32),
                "landmark_coord": landmark_coord.astype(np.float32),
                "landmark_class": landmark_class.astype(np.int64),
                "name": Path(scan_path).stem,
                "full_path": scan_path,
            }
        )
        input_dict = collate_fn([data_dict])
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.no_grad():
            output_dict = model(input_dict)
        torch.cuda.synchronize()
        forward_time += time.perf_counter() - forward_start

        name = input_dict["name"][0]
        rows.extend(prediction_rows(name, output_dict["pred_landmarks"][0], class_names))
        proc_time += time.perf_counter() - proc_start
        print(f"[{idx + 1}/{num_scans}] {name}")
    total_time = time.perf_counter() - start
    print(
        f"Inference on {num_scans} scans: {total_time:.2f}s total ({total_time / num_scans:.3f}s/scan)\n"
        f"  I/O (mesh + GT load):   {io_time:.2f}s ({io_time / num_scans:.3f}s/scan)\n"
        f"  processing (excl. I/O): {proc_time:.2f}s ({proc_time / num_scans:.3f}s/scan)\n"
        f"  model forward:          {forward_time:.2f}s ({forward_time / num_scans:.3f}s/scan)"
    )

    output_dir = args.output_dir or os.path.join(cfg.save_path, f"eval_{args.split}")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions.csv")
    write_predictions_csv(rows, csv_path)
    print(f"Wrote {len(rows)} predictions to {csv_path}")

    if args.gold is not None:
        keys = {Path(p).stem for p in dataset.data_list}
        gold = subset_gold(load_gold(args.gold), keys)
    else:
        gold = gold_from_dataset(dataset, class_names)
    scores = challenge_score(rows, gold)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(scores, f, indent=4)
    for key, value in scores.items():
        print(f"{key}: {value:.4f}")
    print(f"Wrote scores to {results_path}")


if __name__ == "__main__":
    main()
