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
        [--output-dir /path/to/output] [--project-to-surface] \
        [--options model.nms_radius=[0.75,0.75,0.75,0.75,0.75,0.75]]

--options overrides config values without editing the config file (same
KEY=VALUE syntax and dotted keys as tools/train.py), e.g. to try a different
nms_radius against an already-trained checkpoint without touching the config
that was actually used for training.
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
    project_to_surface,
    subset_gold,
    write_predictions_csv,
)
from models import build_model
from utils.config import Config, DictAction


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
    parser.add_argument(
        "--project-to-surface", action="store_true",
        help="snap predictions to the nearest mesh vertex",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="attach a debugpy listener on localhost:5678 before running "
        "(same convention as tools/train.py)",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction,
        help="override config values, e.g. --options model.nms_radius="
        "[0.75,0.75,0.75,0.75,0.75,0.75] (same syntax as tools/train.py)",
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
    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5678))
    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
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
        coord, normal = dataset._load_mesh(scan_path)
        mesh_vertices = coord  # full-resolution, kept for --project-to-surface
        landmark_coord, landmark_class = dataset._load_landmarks(scan_path)
        io_time += time.perf_counter() - load_start

        # mirrors Teeth3DLandmarkDataset.get_data from here on
        proc_start = time.perf_counter()
        sample_idx = dataset._sample_indices(coord)
        coord, normal = coord[sample_idx], normal[sample_idx]
        data_dict = dataset.transform(
            {
                "coord": coord.astype(np.float32),
                "normal": normal.astype(np.float32),
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
        pred = output_dict["pred_landmarks"][0]
        if args.project_to_surface:
            pred = {**pred, "coord": project_to_surface(pred["coord"].cpu().numpy(), mesh_vertices)}
        rows.extend(prediction_rows(name, pred, class_names))
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
