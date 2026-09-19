#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/yyl/lerobot_043_fsdp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
# The CLI value is the global batch. The yyl overlay divides it evenly across two FSDP ranks.
export LEROBOT_GLOBAL_BATCH_SIZE=1

/home/yyl/anaconda3/envs/lerobot/bin/accelerate launch \
  --config_file=/home/yyl/lerobot_043_fsdp/accelerate_fsdp_2gpu.yaml \
  -m lerobot.scripts.lerobot_train \
  --policy.path=/home/yyl/models/smolvla_base \
  --policy.vlm_model_name=/home/yyl/models/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset.repo_id=RITAHuang/smolvla_stacks_0_49_merged \
  --dataset.root=/home/yyl/.cache/huggingface/lerobot/RITAHuang/smolvla_stacks_0_49_merged \
  --batch_size=64 \
  --num_workers=12 \
  --steps=10000 \
  --output_dir=/home/yyl/lerobot_outputs/smolvla/smolvla_stacks_0_49_merged_fsdp2_run1 \
  --job_name=smolvla_stacks_0_49_merged_fsdp2_train \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --save_freq=5000
