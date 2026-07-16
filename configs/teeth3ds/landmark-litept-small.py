"""Point-based 3D landmark detection on FPS-sampled Teeth3DS scans."""

_base_ = ["../_base_/default_runtime.py"]

num_classes = 6
ignore_index = -1
num_points = 8192
grid_size = 0.01

batch_size = 2
batch_size_val = 1
batch_size_test = 1
num_worker = 8
# keep mix_prob at 0: the CutMix path in point_collate_fn does not remap
# landmark_offset, so mixing would silently misalign landmark ground truth
mix_prob = 0
empty_cache = False
enable_amp = True

enable_wandb = True
wandb_project = "bracket_point_prediction"
save_path = "exp/teeth3ds/landmark-litept-small"

model = dict(
    type="LandmarkDetector",
    num_classes=num_classes,
    backbone_out_channels=72,
    class_names=("Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint", "Cusp"),
    # positive_radius / smooth_l1_beta / nms_radius are in original coordinate
    # units (mm); the model converts per sample via coord_scale
    positive_radius=2.0,
    smooth_l1_beta=1.0,
    lambda_coord=1.0,
    lambda_focal=1.0,
    score_threshold=0.5,
    nms_radius=2.0,
    max_predictions_per_class=32,
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
)

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

dataset_type = "Teeth3DLandmarkDataset"
data_root = "/homes/mlugli/BracketPrediction/Teeth3DS/original_data"
fold = "/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_test_split_original"
feat_keys = ["coord"]
collect_keys = [
    "coord",
    "grid_coord",
    "landmark_coord",
    "landmark_class",
    "coord_center",
    "coord_scale",
    "name",
    "full_path",
]
offset_keys = dict(offset="coord", landmark_offset="landmark_coord")

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=["Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint", "Cusp"],
    train=dict(
        type=dataset_type,
        split="train",
        fold=fold,
        data_root=data_root,
        num_points=num_points,
        ignore_index=ignore_index,
        transform=[
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="z", p=0.5),
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-0.1, 0.1], axis="y", p=0.5),
            dict(type="NormalizeCoord"),
            dict(type="GridCoord", grid_size=grid_size),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=collect_keys,
                offset_keys_dict=offset_keys,
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        fold=fold,
        data_root=data_root,
        num_points=num_points,
        deterministic_fps=True,
        ignore_index=ignore_index,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="GridCoord", grid_size=grid_size),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=collect_keys,
                offset_keys_dict=offset_keys,
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
        num_points=num_points,
        deterministic_fps=True,
        ignore_index=ignore_index,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="GridCoord", grid_size=grid_size),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=collect_keys,
                offset_keys_dict=offset_keys,
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LandmarkEvaluator", match_distance_threshold=3.0),
    # gold_path=None derives GT from the splits' __kpt.json files;
    # set it to a gold_standard.pkl to score against a fixed pickle instead
    dict(type="LandmarkChallengeScorer", splits=("val", "test"), gold_path=None),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(type="LandmarkTester", verbose=True)
