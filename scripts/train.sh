#!/bin/bash
#SBATCH --job-name=LitePT_IOS_landmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=8:00:00
#SBATCH --mem=30GB
#SBATCH --output=logs/landmark_train_%j.out
#SBATCH --error=logs/landmark_train_%j.err

cd /homes/mlugli/LitePT || exit 1
mkdir -p logs

source /homes/mlugli/LitePT/.venv/bin/activate
export PYTHONPATH=./

CONFIG="/homes/mlugli/LitePT/configs/teeth3ds/landmark-litept-small.py"
EXP_NAME="LitePT_landmark_100e_fps8192_2"
NUM_GPU=1

python tools/train.py \
  --config-file ${CONFIG} \
  --num-gpus ${NUM_GPU} \
  --options save_path=exp/landmark/${EXP_NAME} batch_size=4 batch_size_val=1 batch_size_test=1 epoch=100 eval_epoch=100
