export HF_TOKEN=TODO
export CACHE_DIR=TODO
export ATTN_IMPLEMENTATION='sda'

CUDA_DEVICE_ID=4

model_name="google/gemma-3-27b-it"
model_pretty_name="gemma27b"
DATASETS=(
  # aime2025
  # aime2024
  # amc23
  minerva
  # math500
  # olympiad
)

CUDA_DEVICE_ID=4
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python main.py --dataset_idx 0