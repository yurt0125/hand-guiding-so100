# SmolVLA 训练进度

更新时间：2026-09-16（Asia/Shanghai）

| 任务 | 数据集 | 状态 | 说明 |
|---|---|---|---|
| banana | `smolvla_banana_0_49_merged` | 已完成 | 10,000 steps；最终 loss 约 0.011；远端输出 `smolvla_banana_0_49_merged_run1`。 |
| blocks | `smolvla_blocks_0_49_merged` | 已完成 | 10,000 steps；最终 loss 约 0.012；远端输出 `smolvla_blocks_0_49_merged_run1`。 |
| hanging | `smolvla_hanging_0_49_merged` | 已完成 | 10,000 steps；最终 loss 约 0.014；远端输出 `smolvla_hanging_0_49_merged_run1`。 |
| pull | `smolvla_pull_0_49_merged` | 已完成 | 10,000 steps；最终 loss 约 0.013；已保存 checkpoint，远端输出 `smolvla_pull_0_49_merged_run1`。 |
| push | `smolvla_push_0_49_merged` | 已完成 | 10,000 steps；最终 loss 约 0.013；已保存 checkpoint，远端输出 `smolvla_push_0_49_merged_run1`。 |
| stacks | `smolvla_stacks_0_49_merged` | 已完成 | 10,000 steps；最终 loss 约 0.014；已保存 checkpoint，远端输出 `smolvla_stacks_0_49_merged_run1`。 |

所有任务使用官方 LeRobot 0.4.3 SmolVLA 训练入口：batch size 64、12 workers、10,000 steps、每 5,000 steps 保存一次、禁用 W&B，且使用服务器本地 SmolVLA / SmolVLM 权重。

## 修正任务文本后的 run2

原始数据和全部 `run1` 均保留不变。以下数据集为独立副本，仅修正 `meta/tasks.parquet` 与 episode 的任务文本：

| 任务 | run2 数据集 | 修正后的任务文本 | 当前状态 |
|---|---|---|---|
| blocks | `smolvla_blocks_0_49_merged_taskfix` | `Put the green block in the basket` | yyl2 GPU 5 正在训练；已通过 step 200。输出目标 `smolvla_blocks_0_49_merged_run2`。 |
| push | `smolvla_push_0_49_merged_taskfix` | `Push the green block to the target area` | GPU 4 启动时被外部 SIGKILL（退出码 137）；现已在 GPU 7 从 0 step 全新训练，且每 1,000 step 保存。 |
| pull | `smolvla_pull_0_49_merged_taskfix` | `Pull the storage box` | 已进入 GPU 5 自动队列：blocks 成功后启动。 |

## 非 banana 任务：语义优化与重训安排

`banana` 保持现有语义与 `run1`，不纳入本轮重训。其余任务均保留原始数据集与 `run1`，在确认任务文本后建立独立 `taskfix` 数据集，并以不变的官方训练参数训练为 `run2`。

| 顺序 | 任务 | 目标任务文本 | 安排 |
|---|---|---|---|
| 1 | blocks | `Put the green block in the basket` | 已建 taskfix，GPU 5 的 run2 正在训练。 |
| 2 | push | `Push the green block to the target area` | 已建 taskfix；GPU 4 曾被外部 SIGKILL，现已在 GPU 7 从 0 step 全新训练，每 1,000 step 保存。 |
| 3 | pull | `Pull the storage box` | 已建 taskfix，GPU 5 自动队列将在 blocks 成功后启动。 |
| 4 | stacks | `Stack the blocks` | 已建 taskfix，GPU 5 自动队列将在 blocks 成功后优先启动；每 1,000 step 保存。 |
| 5 | hanging | `Hang the tape` | 已建 taskfix，GPU 5 自动队列将在 stacks 成功后启动。 |
