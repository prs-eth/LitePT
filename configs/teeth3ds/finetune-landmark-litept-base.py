"""Landmark fine-tuning from segmentation pretraining (LitePT-base).

Replicates the grid-search winner (base_lc4_np16384: lambda_coord=4,
num_points=16384, fold with reserved validation patients) but warm-starts the
backbone from the segmentation pretraining checkpoint via ``weight=``.
CheckpointLoader loads non-strictly: every backbone tensor matches, the
pretrain seg_head is dropped and heat_head/offset_head train from scratch.

Fine-tuning recipe: max_lr halved vs. the from-scratch run (0.006 -> 0.003,
blocks 0.0006 -> 0.0003) since the backbone starts from good weights.
"""

_base_ = ["./landmark-litept-base.py"]

save_path = "exp/teeth3ds/finetune-landmark-litept-base"
weight = "exp/teeth3ds/pretrain-semseg-litept-base/model/model_best.pth"

num_points = 16384
fold = "/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_validation_test_split"

model = dict(
    lambda_coord=4.0,
    lambda_focal=1.0,
    nms_radius=1.0,
    score_threshold=0.15,
    max_predictions_per_class=64,
)

optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.05)
scheduler = dict(max_lr=[0.003, 0.0003])
param_dicts = [dict(keyword="block", lr=0.0003)]

data = dict(
    train=dict(num_points=num_points, fold=fold),
    val=dict(num_points=num_points, fold=fold),
    test=dict(num_points=num_points, fold=fold),
)
