# SmolVLA 机械臂标定快照

此目录保存当前用于 SmolVLA 录制、训练后推理和回放的 LeRobot 机械臂标定副本。

- 创建日期：2026-09-19
- 从臂运行时文件：`/home/robot/.cache/huggingface/lerobot/calibration/robots/so100_follower/single_arm.json`
- 主臂运行时文件：`/home/robot/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader/leader.json`
- 从臂源文件修改时间：2026-09-11 17:55:56 +0800
- 主臂源文件修改时间：2026-09-11 17:53:34 +0800

`follower_single_arm.json` 与 `leader.json` 是上述运行时文件的内容副本，**不是手眼标定矩阵**。

后续若重新标定，请先将新的运行时标定文件复制到此目录，并保留当前版本或以日期创建新快照；录制、训练与部署同一 SmolVLA 模型时，应使用同一套主臂和从臂标定。
