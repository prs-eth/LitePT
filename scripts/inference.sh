#!/bin/bash

set -e

source /homes/mlugli/LitePT/.venv/bin/activate
export PYTHONPATH=./

# current best model: small backbone, lc6, 16384 pts, positive_radius=1,
# focal_alpha=0.5/focal_gamma=1, normals on, calibrated nms_radius, no
# project_to_surface (see exp/landmark/small_backbone_search/summary.csv).
# This is the exact config used for training -- left untouched; overrides,
# if needed, go through --options.
CONFIG="exp/landmark/small_backbone_search/configs/small_lc6_np16384_ep100_pr1_fa0.5_fg1_normals_nmscal.py"
EXP_DIR="exp/landmark/small_backbone_search/small_lc6_np16384_ep100_pr1_fa0.5_fg1_normals_nmscal"
# model_last instead of model_best, to compare the two checkpoints
WEIGHT="${EXP_DIR}/model/model_last.pth"
SPLIT="test"
OUTPUT_DIR="${EXP_DIR}/mm_error_model_last"

python tools/eval_landmarks.py \
  --config-file ${CONFIG} \
  --weight ${WEIGHT} \
  --split ${SPLIT} \
  --output-dir ${OUTPUT_DIR}
