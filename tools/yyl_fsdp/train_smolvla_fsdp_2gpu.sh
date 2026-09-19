#!/usr/bin/env bash
set -euo pipefail

task_name=${1:?"Usage: $0 {pull|push|stacks} [run_name]"}
case "$task_name" in
  pull|push|stacks) ;;
  *) echo "Unsupported task: $task_name" >&2; exit 2 ;;
esac

run_name=${2:-"smolvla_${task_name}_0_49_merged_fsdp2_run1"}
dataset_name="smolvla_${task_name}_0_49_merged"
dataset_root="/home/yyl/.cache/huggingface/lerobot/RITAHuang/${dataset_name}"
output_dir="/home/yyl/lerobot_outputs/smolvla/${run_name}"

test -d "$dataset_root"
test -d /home/yyl/models/smolvla_base
test -d /home/yyl/models/SmolVLM2-500M-Video-Instruct
test ! -e "$output_dir"

export PYTHONPATH=/home/yyl/lerobot_043_fsdp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Default to the first two cards; callers may select a different two-card pair.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Keep the public training setting as global batch size 64. The overlay assigns 32 samples to each rank.
export LEROBOT_GLOBAL_BATCH_SIZE=1

exec /home/yyl/anaconda3/envs/lerobot/bin/accelerate launch \
  --main_process_port=0 \
  --config_file=/home/yyl/lerobot_043_fsdp/accelerate_fsdp_2gpu.yaml \
  -m lerobot.scripts.lerobot_train \
  --policy.path=/home/yyl/models/smolvla_base \
  --policy.vlm_model_name=/home/yyl/models/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset.repo_id="RITAHuang/${dataset_name}" \
  --dataset.root="$dataset_root" \
  --batch_size=64 \
  --num_workers=12 \
  --steps=10000 \
  --output_dir="$output_dir" \
  --job_name="${dataset_name}_fsdp2_train" \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --save_freq=5000
