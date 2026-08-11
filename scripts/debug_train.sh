#!/bin/bash

set -e

source /homes/mlugli/LitePT/.venv/bin/activate
export PYTHONPATH=./

CONFIG="configs/teeth3ds/landmark-litept-small.py"
EXP_NAME="LitePT_landmark_debug_fps8192"
NUM_GPU=1

python tools/train.py \
  --debug \
  --config-file ${CONFIG} \
  --num-gpus ${NUM_GPU} \
  --options save_path=exp/landmark/${EXP_NAME} batch_size=1 batch_size_val=1 batch_size_test=1 epoch=100 eval_epoch=10
