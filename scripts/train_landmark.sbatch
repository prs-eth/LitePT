#!/bin/bash
#SBATCH --job-name=AutobondingV2_tooth_loss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --constraint="gpu_2080_11G|gpu_A40_45G|gpu_K80_12G|gpu_L40S_45G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --output=logs/train_landmark_normals_%j.out
#SBATCH --error=logs/train_landmark_normals_%j.err

cd /homes/mlugli/LitePT || exit 1
mkdir -p logs

source /homes/mlugli/LitePT/.venv/bin/activate
export PYTHONPATH=./

python -u tools/train.py \
  --config-file /homes/mlugli/LitePT/configs/teeth3ds/landmark-litept-base-tooth-loss.py \
  --num-gpus 1 \
  --options epoch=500 save_path=exp/teeth3ds/landmark-litept-base-tooth-loss-400e
