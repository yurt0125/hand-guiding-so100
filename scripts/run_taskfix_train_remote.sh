#!/usr/bin/env bash
# Runs on yyl2 inside a tmux session: fixed official LeRobot training setup.
set -euo pipefail

task_name="$1"
gpu_id="$2"
resume_config_path="${3:-}"
dataset="smolvla_${task_name}_0_49_merged_taskfix"
output_dir="/home/yyl/lerobot_outputs/smolvla/smolvla_${task_name}_0_49_merged_run2"
log_dir="/home/yyl/lerobot_outputs/logs"
mkdir -p "$log_dir"

log_file="$log_dir/smolvla_${task_name}_run2.log"
exit_file="$log_dir/smolvla_${task_name}_run2.exit_code"
echo "$(date -Is) task=${task_name} gpu=${gpu_id}" | tee "$log_file"

save_freq=5000
case "$task_name" in
  push|stacks) save_freq=1000 ;;
esac

if test -n "$resume_config_path"; then
  train_args=(
    "--config_path=$resume_config_path"
    "--resume=true"
    "--save_freq=$save_freq"
  )
else
  train_args=(
    --policy.path=/data1/hrd/models/smolvla_base
    --policy.vlm_model_name=/data1/hrd/models/SmolVLM2-500M-Video-Instruct
    --policy.input_features=null
    --policy.output_features=null
    "--dataset.repo_id=RITAHuang/${dataset}"
    "--dataset.root=/home/yyl/.cache/huggingface/lerobot/RITAHuang/${dataset}"
    "--output_dir=$output_dir"
    "--job_name=smolvla_${task_name}_run2"
    --policy.device=cuda
    --batch_size=64
    --num_workers=12
    --wandb.enable=false
    --policy.push_to_hub=false
    --steps=10000
    "--save_freq=$save_freq"
  )
fi

set +e
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu_id" \
/home/yyl/anaconda3/envs/lerobot/bin/lerobot-train "${train_args[@]}" 2>&1 | tee -a "$log_file"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$exit_file"
exit "$status"
