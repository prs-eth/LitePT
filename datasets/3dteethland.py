"""
Point-based 3D landmark dataset for Teeth3DS intra oral scans.
"""

import json
import os
from pathlib import Path

import fpsample
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from datasets.builder import DATASETS
from datasets.defaults import DefaultDataset
from datasets.teeth3ds import DEFAULT_LABEL_MAPPING
from utils.logger import get_root_logger


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

# background (unmapped label, 0) + 16 FDI tooth positions from DEFAULT_LABEL_MAPPING
NUM_TOOTH_CLASSES = len(set(DEFAULT_LABEL_MAPPING.values())) + 1


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
        sampling="fps",
        load_segment=False,
        debug=False,
    ):
        assert sampling in ("fps", "random"), f"Invalid sampling '{sampling}'"
        self.fold = fold
        self.num_points = num_points
        self.deterministic_fps = deterministic_fps or split != "train"
        self.sampling = sampling
        # when True, also loads the Teeth3DS segmentation mask (same __stem__.json
        # convention as IosDatasetTeeth3ds) and derives, for each landmark, the
        # tooth index of its nearest mesh vertex -- used for auxiliary tooth
        # classification supervision in LandmarkDetector (see predict_tooth)
        self.load_segment = load_segment
        self.default_mapping = DEFAULT_LABEL_MAPPING
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

        if self.load_segment:
            self._report_segment_availability()

    def _report_segment_availability(self):
        """Log how many samples in this split have a segmentation mask on disk.

        Runs once at dataset construction (before training starts) so a
        fold that's missing __stem__.json files -- e.g. an unlabeled test
        split -- shows up in the log immediately instead of crashing deep
        into training the first time that sample is drawn.
        """
        logger = get_root_logger()
        missing = []
        for file_rel_path in self.data_list:
            scan_path = os.path.join(self.data_root, file_rel_path)
            json_path = Path(scan_path).with_suffix(".json")
            if not json_path.exists():
                missing.append(file_rel_path)

        total = len(self.data_list)
        found = total - len(missing)
        logger.info(
            f"[{self.split}] segmentation masks: {found}/{total} samples found"
            + (f", {len(missing)} MISSING" if missing else "")
        )
        if missing:
            preview = ", ".join(missing[:10])
            more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            logger.warning(
                f"[{self.split}] missing segmentation mask for: {preview}{more}"
            )
        return missing

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
            return mesh.vertices.astype(np.float32), mesh.vertex_normals.astype(np.float32)
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

    def _load_segment(self, scan_path):
        """Load per-vertex tooth labels, same convention as IosDatasetTeeth3ds."""
        json_path = Path(scan_path).with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        segment = np.array(data["labels"], dtype=np.int32)
        segment[segment <= 28] += 20
        mapped = np.zeros_like(segment)
        for orig, mapped_label in self.default_mapping.items():
            mapped[segment == orig] = mapped_label
        return mapped

    def _nearest_tooth(self, coord, segment, landmark_coord):
        """Tooth label of the mesh vertex nearest to each landmark."""
        if landmark_coord.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        tree = cKDTree(coord)
        _, nearest_idx = tree.query(landmark_coord, k=1)
        return segment[nearest_idx].astype(np.int64)

    def _sample_indices(self, coord):
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

        if self.sampling == "random":
            # uniform over mesh *vertices*, unlike FPS which is uniform over
            # mesh *surface area* -- keeps whatever density the scanner's own
            # triangulation puts near teeth vs. gum/base
            rng = np.random.RandomState(0) if self.deterministic_fps else np.random
            return np.asarray(
                rng.choice(num_src, num_points, replace=False), dtype=np.int64
            )

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

        coord, normal = self._load_mesh(scan_path)
        landmark_coord, landmark_class = self._load_landmarks(scan_path)

        if self.load_segment:
            # nearest-vertex lookup runs against the full mesh (pre-subsampling)
            # so it isn't biased by which vertices FPS/random sampling kept
            segment = self._load_segment(scan_path)
            assert segment.shape[0] == coord.shape[0], (
                f"Segment shape {segment.shape[0]} != coord shape {coord.shape[0]} for {scan_path}"
            )
            landmark_tooth = self._nearest_tooth(coord, segment, landmark_coord)

        sample_idx = self._sample_indices(coord)
        coord = coord[sample_idx]
        normal = normal[sample_idx]

        data_dict = {
            "coord": coord.astype(np.float32),
            "normal": normal.astype(np.float32),
            "landmark_coord": landmark_coord.astype(np.float32),
            "landmark_class": landmark_class.astype(np.int64),
            "name": Path(scan_path).stem,
            "full_path": scan_path,
        }
        if self.load_segment:
            data_dict["landmark_tooth"] = landmark_tooth
        return data_dict

    def prepare_test_data(self, idx):
        # Landmark validation/testing intentionally uses the same simple path as training.
        return self.prepare_train_data(idx)

    def __len__(self):
        if self.debug:
            return 2
        return len(self.data_list) * self.loop
