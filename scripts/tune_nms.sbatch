#!/bin/bash
#SBATCH --job-name=tune_nms_per_class
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=06:00:00
#SBATCH --mem=16GB
#SBATCH --output=logs/tune_nms_per_class_%j.out
#SBATCH --error=logs/tune_nms_per_class_%j.err

# Calibrates nms_radius (val split, GT read from the dataset's own __kpt.json
# files -- do NOT pass --gold here, gold_standard.pkl covers the test split
# only and gives NaN mAP/mAR on val). One forward pass over the 20 val scans,
# then a per-class radius sweep on the cached candidates -- a few minutes.
#
# To confirm the calibrated radii generalize, rerun with --split test
# (gold_standard.pkl is valid there):
#   python tools/tune_nms_per_class.py --config-file ... --weight ... \
#     --split test --gold /homes/mlugli/BracketPrediction/Mlugli/3dteethland_evaluation/gold_standard.pkl

cd /homes/mlugli/LitePT || exit 1
mkdir -p logs

source /homes/mlugli/LitePT/.venv/bin/activate
export PYTHONPATH=./

python -u tools/tune_nms_per_class.py \
  --config-file exp/landmark/preliminary_grid_search/configs/base_lc4_np16384.py \
  --weight exp/landmark/preliminary_grid_search/base_lc4_np16384/model/model_best.pth \
  --split val \
  --fold /homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_validation_test_split \
  --output-csv exp/landmark/preliminary_nms_grid_search/output.csv
