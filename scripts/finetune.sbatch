#!/bin/bash
#SBATCH --job-name=finetune_landmark_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#SBATCH --output=logs/finetune_landmark_base_%j.out
#SBATCH --error=logs/finetune_landmark_base_%j.err

# Requires the pretrain checkpoint referenced by the config:
#   exp/teeth3ds/pretrain-semseg-litept-base/model/model_best.pth
# Launch standalone after pretraining finishes, or chain at submit time:
#   sbatch --dependency=afterok:<pretrain_job_id> scripts/finetune_landmark_base.sbatch

cd /homes/mlugli/LitePT || exit 1
mkdir -p logs

source /homes/mlugli/LitePT/.venv/bin/activate
export PYTHONPATH=./

python tools/train.py \
  --config-file /homes/mlugli/LitePT/configs/teeth3ds/finetune-landmark-litept-base.py \
  --num-gpus 1
