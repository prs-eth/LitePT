"""
Point-based 3D landmark dataset for Teeth3DS intra oral scans.
"""

import json
import os
from pathlib import Path

import fpsample
import numpy as np
import trimesh

from datasets.builder import DATASETS
from datasets.defaults import DefaultDataset


SPLIT_FILE_MAPPING = {
    "train": ["training_lower.txt", "training_upper.txt"],
    "val": ["validation_lower.txt", "validation_upper.txt"],
    "test": ["testing_lower.txt", "testing_upper.txt"],
}

LANDMARK_CLASSES = (
    "Mesial",
    "Distal",
    "InnerPoint",
    "OuterPoint",
    "FacialPoint",
    "Cusp",
)


@DATASETS.register_module()
class Teeth3DLandmarkDataset(DefaultDataset):
    """Teeth3DS scan loader with same-directory ``__kpt.json`` landmarks."""

    def __init__(
        self,
        data_root,
        fold=None,
        split="train",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        num_points=8192,
        deterministic_fps=False,
        debug=False,
    ):
        self.fold = fold
        self.num_points = num_points
        self.deterministic_fps = deterministic_fps or split != "train"
        self.debug = debug
        self.class_to_idx = {name: idx for idx, name in enumerate(LANDMARK_CLASSES)}
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
            ignore_index=ignore_index,
        )

    def get_data_list(self):
        if self.fold is None:
            file_list = [
                os.path.relpath(os.path.join(root, f), self.data_root)
                for root, _, files in os.walk(self.data_root)
                for f in files
                if f.endswith((".obj", ".stl"))
            ]
            print(f"Loaded {len(file_list)} samples from {self.data_root}")
            return file_list

        fold_dir = Path(self.fold)
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {self.fold}")

        split_files = []
        for split in self.split.split():
            if split not in SPLIT_FILE_MAPPING:
                raise ValueError(
                    f"Invalid split '{split}'. Must be one of {list(SPLIT_FILE_MAPPING)}"
                )
            split_files += SPLIT_FILE_MAPPING[split]

        file_list = []
        for split_file in split_files:
            split_file_path = fold_dir / split_file
            if not split_file_path.exists():
                fallback_name = split_file.replace("validation_", "testing_", 1)
                fallback_path = fold_dir / fallback_name
                if split_file.startswith("validation_") and fallback_path.exists():
                    print(
                        f"Warning: Split file not found: {split_file_path}. "
                        f"Using {fallback_path} for validation."
                    )
                    split_file_path = fallback_path
                else:
                    print(f"Warning: Split file not found: {split_file_path}")
                    continue

            arch = "lower" if "lower" in split_file else "upper"
            with open(split_file_path) as f:
                for line in f:
                    patient_id = line.strip().split("_")[0]
                    if not patient_id:
                        continue
                    for ext in (".obj", ".stl"):
                        scan_path = (
                            Path(self.data_root) / arch / patient_id / f"{patient_id}_{arch}{ext}"
                        )
                        if scan_path.exists():
                            file_list.append(
                                str(Path(arch) / patient_id / f"{patient_id}_{arch}{ext}")
                            )
                            break
                    else:
                        print(f"Warning: File not found: {scan_path}")

        print(f"Loaded {len(file_list)} samples from fold {self.fold}, split {self.split}")
        return file_list

    def _load_mesh(self, scan_path):
        try:
            mesh = trimesh.load_mesh(scan_path, force="mesh")
            return mesh.vertices.astype(np.float32)
        except Exception:
            print(f"Couldn't load sample {scan_path}")
            raise

    def _load_landmarks(self, scan_path):
        kpt_path = Path(scan_path).with_name(f"{Path(scan_path).stem}__kpt.json")
        if not kpt_path.exists():
            raise FileNotFoundError(f"Landmark file not found: {kpt_path}")

        with open(kpt_path) as f:
            data = json.load(f)

        coords = []
        classes = []
        for obj in data.get("objects", []):
            cls_name = obj.get("class")
            if cls_name not in self.class_to_idx:
                continue
            coords.append(obj["coord"])
            classes.append(self.class_to_idx[cls_name])

        if len(coords) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return np.asarray(coords, dtype=np.float32), np.asarray(classes, dtype=np.int64)

    def _fps_indices(self, coord):
        num_points = int(self.num_points)
        num_src = coord.shape[0]
        # fallback / emergency
        if num_points <= 0 or num_src == num_points:
            return np.arange(num_src, dtype=np.int64)
        if num_src < num_points:
            if self.deterministic_fps:
                extra = np.arange(num_points - num_src, dtype=np.int64) % num_src
            else:
                extra = np.random.choice(num_src, num_points - num_src, replace=True).astype(
                    np.int64
                )
            return np.concatenate([np.arange(num_src, dtype=np.int64), extra])
        # returns indices of selected points; fpsample picks a random start
        # index unless one is given, so pin it for deterministic val/test
        points_np = np.ascontiguousarray(coord, dtype=np.float32)
        start_idx = 0 if self.deterministic_fps else None
        selected = fpsample.bucket_fps_kdline_sampling(
            points_np, num_points, h=3, start_idx=start_idx
        )
        return np.asarray(selected, dtype=np.int64)

    def get_data(self, idx, testing=False):
        file_rel_path = self.data_list[idx % len(self.data_list)]
        scan_path = os.path.join(self.data_root, file_rel_path)

        coord = self._load_mesh(scan_path)
        landmark_coord, landmark_class = self._load_landmarks(scan_path)
        fps_idx = self._fps_indices(coord)
        coord = coord[fps_idx]

        return {
            "coord": coord.astype(np.float32),
            "landmark_coord": landmark_coord.astype(np.float32),
            "landmark_class": landmark_class.astype(np.int64),
            "name": Path(scan_path).stem,
            "full_path": scan_path,
        }

    def prepare_test_data(self, idx):
        # Landmark validation/testing intentionally uses the same simple path as training.
        return self.prepare_train_data(idx)

    def __len__(self):
        if self.debug:
            return 2
        return len(self.data_list) * self.loop
