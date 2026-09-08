"""landmark-litept-base-nms-calibrated.py plus surface normals as an extra
input feature (3 -> 6 channels): each point's mesh vertex normal, concatenated
after coord. Requires retraining -- in_channels is baked into the backbone's
first projection layer.
"""

_base_ = ["./landmark-litept-base-nms-calibrated.py"]

save_path = "exp/teeth3ds/landmark-litept-base-normals"

feat_keys = ["coord", "normal"]

model = dict(backbone=dict(in_channels=6))

_collect_keys = [
    "coord",
    "grid_coord",
    "landmark_coord",
    "landmark_class",
    "coord_center",
    "coord_scale",
    "name",
    "full_path",
]
_offset_keys = dict(offset="coord", landmark_offset="landmark_coord")

_train_transform = [
    dict(type="RandomRotate", angle=[-0.1, 0.1], axis="z", p=0.5),
    dict(type="RandomRotate", angle=[-0.1, 0.1], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-0.1, 0.1], axis="y", p=0.5),
    dict(type="NormalizeCoord"),
    dict(type="GridCoord", grid_size=0.01),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=_collect_keys,
        offset_keys_dict=_offset_keys,
        feat_keys=feat_keys,
    ),
]

_eval_transform = [
    dict(type="NormalizeCoord"),
    dict(type="GridCoord", grid_size=0.01),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=_collect_keys,
        offset_keys_dict=_offset_keys,
        feat_keys=feat_keys,
    ),
]

data = dict(
    train=dict(transform=_train_transform),
    val=dict(transform=_eval_transform),
    test=dict(transform=_eval_transform),
)
