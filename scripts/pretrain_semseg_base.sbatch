#!/bin/bash
#SBATCH --job-name=pretrain_semseg_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --output=logs/pretrain_semseg_base_%j.out
#SBATCH --error=logs/pretrain_semseg_base_%j.err

cd /homes/mlugli/LitePT || exit 1
mkdir -p logs

source /homes/mlugli/LitePT/.venv/bin/activate
export PYTHONPATH=./

python tools/train.py \
  --config-file /homes/mlugli/LitePT/configs/teeth3ds/pretrain-semseg-litept-base.py \
  --num-gpus 1
