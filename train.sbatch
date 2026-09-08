#!/bin/bash
#SBATCH --job-name=PTLite_IOS_segmentation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create necessary directories

# Activate your environment
source /homes/mlugli/LitePT/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=./

# Configuration
CONFIG="/homes/mlugli/LitePT/configs/teeth3ds/semseg-litpt-small.py"
EXP_NAME="LitePT_segmentation_100e_batched"
NUM_GPU=1

# Training command
python tools/train.py --config-file ${CONFIG} --num-gpus ${NUM_GPU} --options save_path=exp/semseg/${EXP_NAME} batch_size=16 epoch=100 eval_epoch=50
