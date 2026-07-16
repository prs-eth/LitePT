"""
Intra Oral Scan segmentator dataset.
Author: Matteo Lugli (283122@studenti.unimore.it)
Please cite our work if the code is helpful to you.
"""

import os
import json
import numpy as np
import trimesh
from copy import deepcopy
from pathlib import Path

from datasets.builder import DATASETS
from datasets.transform import Compose
from datasets.defaults import DefaultDataset

SPLIT_FILE_MAPPING = {
    'train': ['training_lower.txt', 'training_upper.txt'],
    'val':   ['validation_lower.txt', 'validation_upper.txt'],
    'test':  ['testing_lower.txt', 'testing_upper.txt'],
}

DEFAULT_LABEL_MAPPING = {
    48: 1,  47: 2,  46: 3,
    45: 4,  44: 5,  43: 6,
    42: 7,  41: 8,  31: 9,
    32: 10, 33: 11, 34: 12,
    35: 13, 36: 14, 37: 15,
    38: 16,
}

@DATASETS.register_module()
class IosDatasetTeeth3ds(DefaultDataset):
    """Dataset for tooth segmentation from OBJ files (Teeth3DS format)."""

    def __init__(
        self,
        data_root,
        fold=None,
        split="train",
        transform=None,
        test_mode=False,
        test_cfg=None,
        load_segment=True,
        loop=1,
        ignore_index=0,
        debug=False,
    ):
        self.fold = fold
        self.debug = debug
        self.load_segment = load_segment
        self.default_mapping = DEFAULT_LABEL_MAPPING
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
            ignore_index=ignore_index,
        )

        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]

    def get_data_list(self):
        """Load list of OBJ samples.
        
        If fold is None (inference mode), walks data_root and returns all .obj files.
        Otherwise, reads the appropriate split text files from the fold directory.
        """
        if self.fold is None:
            file_list = [
                os.path.relpath(os.path.join(root, f), self.data_root)
                for root, _, files in os.walk(self.data_root)
                for f in files if f.endswith(('.obj', '.stl'))
            ]
            print(f"Loaded {len(file_list)} samples from {self.data_root} (inference mode)")
            return file_list

        fold_dir = Path(self.fold)
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {self.fold}")

        split_files = []
        for s in self.split.split():
            if s not in SPLIT_FILE_MAPPING:
                raise ValueError(f"Invalid split '{s}'. Must be one of {list(SPLIT_FILE_MAPPING)}")
            split_files += SPLIT_FILE_MAPPING[s]

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

            arch = 'lower' if 'lower' in split_file else 'upper'
            with open(split_file_path) as f:
                for line in f:
                    patient_id = line.strip().split('_')[0]
                    if not patient_id:
                        continue
                    for ext in (".obj", ".stl"):
                        obj_path = Path(self.data_root) / arch / patient_id / f"{patient_id}_{arch}{ext}"
                        if obj_path.exists():
                            file_list.append(str(Path(arch) / patient_id / f"{patient_id}_{arch}{ext}"))
                            break
                    else:
                        print(f"Warning: File not found: {obj_path}")

        print(f"Loaded {len(file_list)} samples from fold {self.fold}, split {self.split}")
        return file_list

    def _load_obj(self, obj_path):
        arch = "lower" if "lower" in obj_path else "upper"
        try:
            if Path(obj_path).suffix == ".stl":
                mesh = trimesh.load(obj_path, force='mesh')
            else:
                mesh = trimesh.load_mesh(obj_path, process=False)
            return mesh.vertices.astype(np.float32), mesh.vertex_normals.astype(np.float32)
        except Exception:
            print(f"Couldn't load sample {obj_path}")
            raise

    def _load_json(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        segment = np.array(data['labels'], dtype=np.int32)
        segment[segment <= 28] += 20
        mapped = np.zeros_like(segment)
        for orig, mapped_label in self.default_mapping.items():
            mapped[segment == orig] = mapped_label
        return mapped

    def get_data(self, idx, testing=False):
        file_rel_path = self.data_list[idx % len(self.data_list)]
        obj_path = os.path.join(self.data_root, file_rel_path)
        json_path = str(Path(obj_path).with_suffix(".json"))

        coord, normal = self._load_obj(obj_path)

        if self.load_segment:
            segment = self._load_json(json_path)
            assert segment.shape[0] == coord.shape[0], (
                f"Segment shape {segment.shape[0]} != coord shape {coord.shape[0]} for {obj_path}"
            )
        else:
            segment = np.zeros(coord.shape[0], dtype=np.int32)

        return {
            "coord": coord,
            "normal": normal,
            "name": Path(obj_path).stem,
            "full_path": obj_path,
            "segment": segment,
        }

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx, testing=True)
        data_dict = self.transform(data_dict)

        result_dict = dict(
            segment=data_dict.pop("segment"),
            name=data_dict.get("name"),
            full_path=data_dict.get("full_path"),
        )

        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = [aug(deepcopy(data_dict)) for aug in self.aug_transform]

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                data_part = self.test_crop(data_part) if self.test_crop is not None else [data_part]
                fragment_list += data_part

        result_dict["fragment_list"] = [self.post_transform(f) for f in fragment_list]
        return result_dict

    def __len__(self):
        if self.debug: return 2
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class IosDatasetTeeth3dsCached(IosDatasetTeeth3ds):
    """Teeth3DS inference dataset that serves scans from in-memory cache."""

    def __init__(self, *args, custom_cache=None, **kwargs):
        if custom_cache is None:
            raise ValueError("IosDatasetTeeth3dsCached requires a cache parameter")
        self.custom_cache = custom_cache
        super().__init__(*args, **kwargs)

    def get_data_list(self):
        scan_names = list(self.custom_cache.scan_meshes.keys())
        print(f"Loaded {len(scan_names)} samples from cache")
        return scan_names

    def _load_obj(self, obj_path):
        arch = "lower" if "lower" in obj_path else "upper"
        try:
            cached_mesh = self.custom_cache.get_scan_mesh(Path(obj_path))
            if cached_mesh is not None: mesh = cached_mesh.copy()
            else: mesh = trimesh.load_mesh(obj_path, process=False)
            mesh.merge_vertices()
            return mesh.vertices.astype(np.float32), mesh.vertex_normals.astype(np.float32)
        except Exception:
            print(f"Couldn't load sample {obj_path}")
            raise
