"""Segmentation pretraining for the landmark detector (LitePT-small).

Trains a 17-class (16 FDI pairs + gum) semantic segmentator whose backbone is
weight-compatible with the LandmarkDetector in landmark-litept-small.py, so the
checkpoint can be used to warm-start landmark fine-tuning via
``--options weight=<this run>/model/model_best.pth``.

Derived from semseg-litpt-small.py (kept as reference) with the changes needed
for backbone transfer:
+ coord-only input features (in_channels=3) to match the landmark model;
+ FPS subsampling to num_points + GridCoord (instead of GridSample
  voxelization) so point density / patch statistics match the landmark
  pipeline — keep num_points equal to the landmark run being warm-started;
+ explicit save_path;
+ standard "test" split with labels, no PreciseEvaluator.

Scans are expected in the standard orientation described in
semseg-litpt-small.py; normalization is performed online.

The fold reserves the 10 landmark-validation patients (20 scans) as the
pretraining validation set, so pretraining never trains on patients used for
landmark model selection; everything else (Teeth3DS train + test, 1780 scans)
is training data. There is no test split: pretraining only needs val
monitoring for checkpoint selection.
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------
# Misc settings
# -----------------------------
num_classes = 17  # 16 FDI Indices + Gum
ignore_index = -1
segment_ignore_index = (-1)

batch_size = 8
num_worker = 8
mix_prob = 0
empty_cache = False
enable_amp = True

# -----------------------------
# Wandb settings
# -----------------------------
enable_wandb = True
wandb_project = "bracket_point_prediction"
save_path = "exp/teeth3ds/pretrain-semseg-litept-small"

# -----------------------------
# Model settings
# -----------------------------
# backbone must stay identical to landmark-litept-small.py (including
# in_channels=3: the landmark model feeds coord only) or the checkpoint
# will not load into the LandmarkDetector
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        attn_backend="torch",
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
)

# -----------------------------
# Optimizer & Scheduler
# -----------------------------
epoch = 100
eval_epoch = 10
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# -----------------------------
# Dataset settings
# -----------------------------
dataset_type = "IosDatasetTeeth3ds"
data_root = "/homes/mlugli/BracketPrediction/data/original_data"
feat_keys = ["coord"]
grid_size = 0.01
num_points = 8192  # match the num_points of the landmark run being warm-started
fold = "/homes/mlugli/BracketPrediction/data/splits/Teeth3DS_train_validation_test_pretrain"

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=[
        "Gum", "48-28", "47-27", "46-26", "45-25", "44-24", "43-23", "42-22", "41-21",
        "31-11", "32-12", "33-13", "34-14", "35-15", "36-16", "37-17", "38-18"
    ],
    train=dict(
        type=dataset_type,
        split="train",
        fold=fold,
        data_root=data_root,
        ignore_index=ignore_index,
        num_points=num_points,
        transform=[
            # must enable normalize_coord on original data
            dict(type="NormalizeCoord"),
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="z", p=0.5),
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            dict(type="RandomShift", shift=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
            dict(type="GridCoord", grid_size=grid_size),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "name", "segment"],
                feat_keys=feat_keys)
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        fold=fold,
        data_root=data_root,
        ignore_index=ignore_index,
        num_points=num_points,
        deterministic_fps=True,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="GridCoord", grid_size=grid_size),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "segment", "name"],
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        fold=fold,
        data_root=data_root,
        ignore_index=ignore_index,
        num_points=num_points,
        deterministic_fps=True,
        transform=[
            dict(type="NormalizeCoord"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=feat_keys),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)

# -----------------------------
# Hooks
# -----------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
test = dict(type="SemSegTester")
