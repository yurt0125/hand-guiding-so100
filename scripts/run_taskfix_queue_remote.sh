#!/usr/bin/env bash
# Runs on yyl2. Queue GPU jobs only after the preceding task succeeds.
set -euo pipefail

gpu_id="$1"
shift
tasks=("$@")
log_dir="/home/yyl/lerobot_outputs/logs"
launcher="/home/yyl/lerobot_outputs/run_taskfix_train_remote.sh"
queue_log="$log_dir/smolvla_run2_gpu${gpu_id}_queue.log"
mkdir -p "$log_dir"

is_training() {
  pgrep -f -- "--job_name=smolvla_${1}_run2" >/dev/null
}

wait_for_success() {
  local task_name="$1"
  local exit_file="$log_dir/smolvla_${task_name}_run2.exit_code"
  while is_training "$task_name"; do
    sleep 60
  done
  if test -f "$exit_file" && test "$(cat "$exit_file")" = "0"; then
    return 0
  fi
  echo "$(date -Is) ${task_name} did not finish successfully; queue stops." | tee -a "$queue_log"
  return 1
}

for task_name in "${tasks[@]}"; do
  if is_training "$task_name"; then
    echo "$(date -Is) waiting for active ${task_name} on GPU ${gpu_id}" | tee -a "$queue_log"
    wait_for_success "$task_name"
    continue
  fi

  echo "$(date -Is) starting ${task_name} on GPU ${gpu_id}" | tee -a "$queue_log"
  rm -f "$log_dir/smolvla_${task_name}_run2.exit_code"
  "$launcher" "$task_name" "$gpu_id"
  wait_for_success "$task_name"
done

echo "$(date -Is) queue completed successfully." | tee -a "$queue_log"
