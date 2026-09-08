"""Landmark inference on a folder of STL scans.

Recursively finds .stl files under the input folder and, for each scan, writes
<stem>_landmarks.json ({class_name: [[x, y, z], ...]}, sorted by descending
score) and <stem>_landmarks.ply (predicted landmarks as a point cloud, one
color per class) either into an output folder mirroring the input structure or,
with --inplace, next to each scan.

Scans with a different orientation than the training data can be pre-rotated
with --rotate RX RY RZ (degrees, applied about the fixed x, y, z axes in that
order); the inverse rotation is applied to the predictions, so outputs are
always in the input scan's reference frame.

--project-to-surface snaps each prediction to its nearest mesh vertex. GT
landmarks sit essentially on the surface while predicted offsets don't, so
this trims the off-surface component of the error (empirically verified on
val: a few points of AP gain concentrated at the sub-mm distance thresholds,
no measurable downside).

Usage:
    python tools/infer_landmarks.py IN_DIR --output OUT_DIR [--rotate 90 0 0]
    python tools/infer_landmarks.py IN_DIR --inplace --project-to-surface
"""

import argparse
import json
from pathlib import Path

import fpsample
import numpy as np
import torch
import trimesh

from datasets.transform import Compose
from metrics.landmark import project_to_surface
from models.builder import build_model
from utils.config import Config

GRID_WINNER_ROOT = "/work/grana_maxillo/Mlugli/models/Landmarks/landmark/preliminary_grid_search"
DEFAULT_CONFIG = f"{GRID_WINNER_ROOT}/configs/base_lc4_np16384.py"
DEFAULT_WEIGHT = f"{GRID_WINNER_ROOT}/base_lc4_np16384/model/model_best.pth"

CLASS_COLORS = {
    "Mesial": (230, 25, 75),
    "Distal": (0, 130, 200),
    "InnerPoint": (60, 180, 75),
    "OuterPoint": (255, 225, 25),
    "FacialPoint": (240, 50, 230),
    "Cusp": (70, 240, 240),
}


def rotation_matrix(degrees):
    rx, ry, rz = np.radians(degrees)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (mz @ my @ mx).astype(np.float32)


def fps_subsample(coord, num_points):
    num_src = coord.shape[0]
    if num_src == num_points:
        return coord
    if num_src < num_points:
        extra = np.arange(num_points - num_src, dtype=np.int64) % num_src
        return coord[np.concatenate([np.arange(num_src, dtype=np.int64), extra])]
    idx = fpsample.bucket_fps_kdline_sampling(
        np.ascontiguousarray(coord), num_points, h=3, start_idx=0
    )
    return coord[np.asarray(idx, dtype=np.int64)]


def load_model(cfg, weight, device):
    model = build_model(cfg.model)
    state_dict = torch.load(weight, map_location="cpu")["state_dict"]
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, pipeline, coord, device):
    data = pipeline({"coord": coord})
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    pred = model(data)["pred_landmarks"][0]
    return (
        pred["coord"].cpu().numpy(),
        pred["class"].cpu().numpy(),
        pred["score"].cpu().numpy(),
    )


def write_outputs(out_dir, stem, coords, classes, scores, class_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-scores)
    coords, classes = coords[order], classes[order]

    landmarks = {name: [] for name in class_names}
    for coord, cls in zip(coords, classes):
        landmarks[class_names[cls]].append([float(c) for c in coord])
    with open(out_dir / f"{stem}_landmarks.json", "w") as f:
        json.dump(landmarks, f, indent=2)

    if len(coords):
        colors = np.array([CLASS_COLORS[class_names[c]] for c in classes], dtype=np.uint8)
        trimesh.PointCloud(coords, colors=colors).export(out_dir / f"{stem}_landmarks.ply")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="folder containing .stl scans")
    parser.add_argument("--output", type=Path, help="output folder (mirrors input structure)")
    parser.add_argument("--inplace", action="store_true", help="write results next to each scan")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="model config file")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, help="model checkpoint")
    parser.add_argument(
        "--rotate", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"),
        help="rotation in degrees about x, y, z applied before inference; "
             "predictions are rotated back to the input frame",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=None,
        help="override the config's decode threshold (e.g. 0.5 for a realistic output)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="skip scans whose landmarks json already exists (resume an interrupted run)",
    )
    parser.add_argument(
        "--project-to-surface", action="store_true",
        help="snap predictions to the nearest mesh vertex",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.inplace == (args.output is not None):
        parser.error("exactly one of --output / --inplace is required")

    cfg = Config.fromfile(args.config)
    if args.score_threshold is not None:
        cfg.model.score_threshold = args.score_threshold
    model = load_model(cfg, args.weight, args.device)
    class_names = list(model.class_names)
    num_points = cfg.data["val"]["num_points"]
    pipeline = Compose([
        dict(type="NormalizeCoord"),
        dict(type="GridCoord", grid_size=cfg.grid_size),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "coord_center", "coord_scale"),
            feat_keys=cfg.feat_keys,
        ),
    ])
    rot = rotation_matrix(args.rotate)

    scans = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in  [".stl", ".obj"] )
    if not scans:
        raise SystemExit(f"no .stl files found under {args.input}")

    for i, scan_path in enumerate(scans, 1):
        out_dir = scan_path.parent if args.inplace \
            else args.output / scan_path.parent.relative_to(args.input)
        if args.skip_existing and (out_dir / f"{scan_path.stem}_landmarks.json").exists():
            print(f"[{i}/{len(scans)}] {scan_path.relative_to(args.input)}: skipped (exists)")
            continue
        mesh = trimesh.load_mesh(scan_path, force="mesh")
        coord = fps_subsample(mesh.vertices.astype(np.float32), num_points)
        coords, classes, scores = predict(model, pipeline, coord @ rot.T, args.device)
        coords = coords @ rot  # back to the input scan's frame
        if args.project_to_surface:
            coords = project_to_surface(coords, mesh.vertices)

        write_outputs(out_dir, scan_path.stem, coords, classes, scores, class_names)
        print(f"[{i}/{len(scans)}] {scan_path.relative_to(args.input)}: {len(coords)} landmarks")


if __name__ == "__main__":
    main()
