"""landmark-litept-base-nms-calibrated.py plus auxiliary tooth-index
supervision: for each point matched to a ground-truth landmark, also classify
the tooth (FDI position, background + 16 teeth) of that landmark's nearest
mesh vertex in the Teeth3DS segmentation mask. Requires retraining -- adds a
new tooth_head on top of the backbone features.

Segmentation masks (__stem__.json) are only available for the train/val folds
of this split -- the challenge test fold has none on disk (confirmed via
Teeth3DLandmarkDataset's segment-availability report: 220/220 train, 20/20
val, 0/100 test). The test split is therefore built with load_segment=False;
LandmarkDetector skips the tooth loss term per-batch whenever a batch has no
'landmark_tooth', so evaluation on test still runs, just without that signal.
"""

_base_ = ["./landmark-litept-base-nms-calibrated.py"]

save_path = "exp/teeth3ds/landmark-litept-base-tooth-loss"

predict_tooth = True
num_tooth_classes = 17  # background (0) + 16 FDI tooth positions

model = dict(
    predict_tooth=predict_tooth,
    num_tooth_classes=num_tooth_classes,
    lambda_tooth=1.0,
)

_feat_keys = ["coord"]

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
_collect_keys_with_tooth = _collect_keys + ["landmark_tooth"]
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
        keys=_collect_keys_with_tooth,
        offset_keys_dict=_offset_keys,
        feat_keys=_feat_keys,
    ),
]

_eval_transform_with_tooth = [
    dict(type="NormalizeCoord"),
    dict(type="GridCoord", grid_size=0.01),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=_collect_keys_with_tooth,
        offset_keys_dict=_offset_keys,
        feat_keys=_feat_keys,
    ),
]

# test fold has no segmentation masks on disk -- collect without landmark_tooth
_eval_transform_no_tooth = [
    dict(type="NormalizeCoord"),
    dict(type="GridCoord", grid_size=0.01),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=_collect_keys,
        offset_keys_dict=_offset_keys,
        feat_keys=_feat_keys,
    ),
]

data = dict(
    train=dict(load_segment=predict_tooth, transform=_train_transform),
    val=dict(load_segment=predict_tooth, transform=_eval_transform_with_tooth),
    test=dict(load_segment=False, transform=_eval_transform_no_tooth),
)
