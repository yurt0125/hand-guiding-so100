# Hand Guiding SO100 Pipeline

SO100 机械臂手部示教项目，包含从标定、RGB-D 采集、轨迹提取、可视化验收到执行回放的完整流程。

## Project Structure

```text
project/
  app/            # 实时运行入口（键盘/视觉/回放）
  common/         # 运动学、控制与共享模块
  calibration/    # 手眼标定工具与输出
  pipelines/      # 离线流程脚本（record/extract/visualize/export）
  data/           # 数据目录（raw_episodes/hamer_outputs/replay_csv/...）
  artifacts/      # 可视化产物与调试视频
  legacy/         # 历史脚本存档
```

## Quick Start

### 0) 检查摄像头

```bash
PYTHONPATH=third-party/lerobot/src python third-party/lerobot/src/lerobot/scripts/lerobot_find_cameras.py \
  --output-dir calibration/inputs/camera_check \
  --record-time-s 1
```

说明：
- 该命令会把检测到的摄像头画面保存到 `calibration/inputs/camera_check/`。
- 你直接看保存的图片来判断哪个是顶部 RGB、哪个是 D435 RGB。

### 1) 手眼标定

第一步：采点求解外参

```bash
python calibration/1_dual_camera_robot_click_calibration.py \
  --robot-port /dev/ttyACM0 \
  --min-pairs 5
```

标定矩阵主文件：
- `calibration/outputs/cam2base_latest.json`

说明：
- 你移动机械臂到目标位后，在 `D435 RGB` 点击同一末端点，按 `G` 采样。
- 脚本会自动记录当前机械臂末端 `robot_x/y/z` 与 D435 相机3D坐标。
- 建议采样 `16~24` 个点（最少 `12` 个），覆盖近/远、左/右、高/低，避免点都在同一平面。
- 按 `P` 保存并在脚本内直接求解。
- 机械臂控制按键与 `python run_pipeline.py --mode keyboard` 保持一致：
  - `W/S` 前后，`A/D` 左右，`Q/E` 上下，`I/K` roll，`J/L` pitch，`C/V` 夹爪
  - 标定脚本直接复用同一控制类（`SO100TeleopViewer + KeyboardController`），不再单独实现一套控臂逻辑
- 采样按键（与控制解耦）：
  - `G` 记录点，`B` 清空当前点击，`P` 保存并求解，`ESC` 退出

第二步：点击精度验收（验证标定）

```bash
python calibration/2_click_to_base_verify.py \
  --transform-json calibration/outputs/cam2base_latest.json \
  --patch-radius 0
```

说明：
- 左键点击 D435 画面中的点，程序会直接打印该点在机械臂 base 坐标系下的位置。

### 2) 采集数据

统一使用：
`pipelines/record/1_lerobot_record_so101_depth.py`

#### A. D435 + RGB（连接机械臂，官方录制流程）

```bash
PYTHONPATH=third-party/lerobot/src:. python pipelines/record/1_lerobot_record_so101_depth.py \
  --robot.disable_torque_on_disconnect=true \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=left \
  --robot.cameras="{
    'd435': {
      'type': 'intelrealsense',
      'serial_number_or_name': '216322074780',
      'width': 640,
      'height': 480,
      'fps': 15,
      'use_depth': true
    },
    'fixed': {
      'type': 'opencv',
      'index_or_path': '/dev/video13',
      'width': 640,
      'height': 480,
      'fps': 15
    }
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader \
  --display_data=false \
  --dataset.repo_id=rita/test \
  --dataset.root=/home/rita/HandGuiding_test/project/data/so101_official9 \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Grab the bananna" \
  --dataset.fps=15
```

#### B. 单 D435（连接机械臂，官方录制流程）

```bash
PYTHONPATH=third-party/lerobot/src:. python pipelines/record/1_lerobot_record_so101_depth.py \
  --robot.disable_torque_on_disconnect=true \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=left \
  --robot.cameras="{
    'd435': {
      'type': 'intelrealsense',
      'serial_number_or_name': '216322074780',
      'width': 640,
      'height': 480,
      'fps': 30,
      'use_depth': true
    }
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader \
  --display_data=true \
  --dataset.repo_id=rita/test \
  --dataset.root=/home/rita/HandGuiding_test/project/data/so101_official10 \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Grab the bananna" \
  --dataset.fps=30
```

#### C. D435 + RGB（不连接机械臂，camera-only）

```bash
PYTHONPATH=third-party/lerobot/src:. python pipelines/record/1_lerobot_record_so101_depth.py \
  --camera-only \
  --robot.cameras="{
    'd435': {'type': 'intelrealsense', 'serial_number_or_name': '216322074780', 'width': 640, 'height': 480, 'fps': 15, 'use_depth': true},
    'fixed': {'type': 'opencv', 'index_or_path': '/dev/video13', 'width': 640, 'height': 480, 'fps': 15}
  }" \
  --dataset.root=/home/rita/HandGuiding_test/project/data/raw \
  --dataset.repo_id=5.3 \
  --display_data=true \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.single_task="camera only collection" \
  --dataset.fps=15
```

#### D. D435（不连接机械臂，camera-only）

```bash
记得设置全局变量名
RUN_NAME = 5.5 或 export RUN_NAME = 5.5

PYTHONPATH=third-party/lerobot/src:. python pipelines/record/1_lerobot_record_so101_depth.py \
  --camera-only \
  --robot.cameras="{
    'd435': {'type': 'intelrealsense', 'serial_number_or_name': '216322074780', 'width': 640, 'height': 480, 'fps': 30, 'use_depth': true}
  }" \
  --dataset.root=/home/rita/HandGuiding_test/project/data/raw/${RUN_NAME} \
  --dataset.repo_id=rita/${RUN_NAME} \
  --display_data=true \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=15 \
  --dataset.push_to_hub=false \
  --dataset.single_task="d435 only collection" \
  --dataset.fps=30
```
可选参数含义：
- `--camera-only`：纯相机模式，不连接主臂/从臂。
- `--display_data=true|false`：是否开启 `rerun` 实时可视化（开了更直观，但更吃性能）。
- `--dataset.root=...`：数据保存目录（建议每次实验用新目录，避免覆盖）。
- `--dataset.repo_id=owner/name`：官方格式要求，必须带 `/`，即使不上传也要满足格式。
- `--dataset.num_episodes=N`：录制 episode 数量。
- `--dataset.episode_time_s=T`：每个 episode 的时长（秒）。
- `--dataset.fps=K`：录制目标帧率（建议先 `15`，稳定后再试 `30`）。
- `--dataset.push_to_hub=false`：仅本地保存，不上传 HuggingFace。
- `--dataset.single_task="..."`：任务文本标签，后续训练可作为任务条件。
- `--robot.cameras="{...}"`：相机配置字典。
- `d435.serial_number_or_name`：D435 序列号或名称（你当前是 `216322074780`）。
- `fixed.index_or_path`：顶部 RGB 设备路径（如 `/dev/video13`）。

说明：
- `--camera-only` 会走官方 `lerobot_record` 主流程，但不连接机械臂。
- 脚本会自动补 `robot.type / robot.id / robot.port / dataset.repo_id` 缺省值（用于通过官方配置检查）。
- 建议先用 `fps=15` 跑稳，再尝试 `fps=30`。
- 录制结束后会自动导出可直接用于轨迹提取的深度文件：
  - `data/raw/<RUN_NAME>/depth/<cam_key>/chunk-xxx/file-xxx/depth.npz`
  - `data/raw/<RUN_NAME>/depth/<cam_key>/chunk-xxx/file-xxx/intrinsics.json`
- 当前流程不再依赖 `depth_packed.mp4 -> 解包`。

深度稳定性复盘（重要）：
- 改进脚本更稳定的核心原因：使用了 `align(depth->color)` + `spatial` + `temporal` + `hole_filling` 的完整深度处理链路，再保存 `uint16` 深度。
- 早期版本不稳定的原因：
  - 直接取原始 depth 帧，未做对齐与滤波，空洞和离群点更多；
  - 走过 `depth_packed.mp4 -> 解包` 链路，深度高低位容易被视频编码/解码扰动；
  - 缺少读取重试时，偶发帧更容易出现 invalid。
- 当前修复点（已落地到 `1_lerobot_record_so101_depth.py`）：
  - 统一改为“采集时直接保存原始 `depth.npz`（uint16）”；
  - 深度读取改为与旧脚本一致：对齐 + 三段滤波 + 重试；
  - 提取与可视化统一从 `data/raw/<RUN_NAME>/depth/...` 读取，不再依赖 packed 深度视频。


#遥操
lerobot-teleoperate     --robot.type=so101_follower     --robot.port=/dev/ttyACM1    --robot.id=left     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM0     --teleop.id=3204 

huggingface-cli login --token "<HUGGING_FACE_TOKEN>" --add-to-git-credential
huggingface-cli whoami

#采集数据
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=left \
    --robot.cameras="{'handeye': {'type':'opencv', 'index_or_path':13, 'width':640, 'height':480, 'fps':30}, 'fixed': {'type':'opencv', 'index_or_path':10, 'width':640, 'height':480, 'fps':30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=3204 \
    --display_data=true \
    --dataset.repo_id=RITAHuang/20260504RL_test \
    --dataset.num_episodes=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=60 \
    --dataset.single_task="Grab the banana" \
    --dataset.fps=30 \
    --dataset.push_to_hub=true \
    --resume=true

rm -rf /home/rita/.cache/huggingface/lerobot/RITAHuang/20260504RL_test

#replay
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=left \
  --dataset.repo_id=RITAHuang/20260504RL_test \
  --dataset.root=/home/rita/.cache/huggingface/lerobot/RITAHuang/20260504RL_test \
  --dataset.episode=0


python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.delete_repo(repo_id="RITAHuang/20260504RL_test", repo_type="dataset")
print("deleted")
PY

#训练
  lerobot-train \
  --dataset.repo_id=RITAHuang/20260504RL_test \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir=/home/rita/HandGuiding_test/project/rl_test \
  --job_name=20260504RL_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=RITAHuang/20260504RL_test_act \
  --policy.push_to_hub=true \
  --steps=40000 \
  --save_freq=8000

#推理
lerobot-record  \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=left \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=3204 \
  --robot.disable_torque_on_disconnect=true \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 10, 'width': 640, 'height': 480, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 12, 'width': 640, 'height': 480, 'fps': 30}}" \
  --display_data=true \
  --dataset.single_task="Grab the banana" \
  --policy.path=/home/rita/HandGuiding_test/project/rl_test/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=RITAHuang/eval_so101 --dataset.push_to_hub=false \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=40 \
  --dataset.num_episodes=10 

rm -rf /home/rita/.cache/huggingface/lerobot/RITAHuang/eval_so101

#HIL数据收集
lerobot-rollout --strategy.type=dagger \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=left \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 12, 'width': 640, 'height': 480, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 10, 'width': 640, 'height': 480, 'fps': 30}}" \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=3204 \
  --policy.path=/home/rita/HandGuiding_test/project/rl_test/checkpoints/last/pretrained_model \
  --dataset.repo_id=RITAHuang/eval_so101 --dataset.push_to_hub=false \
  --dataset.single_task="Grab the banana" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=40 \
  --dataset.num_episodes=10 \
  --interpolation_multiplier=2


### 3) Dyn-HaMR 远程自动处理（上传->推理->回传）

```bash
ssh yyl "echo ok"
```

输出 `ok` 说明免密链路可用。

单视频处理：

```bash
python pipelines/hamer/8_dynhamr_remote_sync.py \
  --video-path /home/rita/D435_record/datasets/d435_hand_demo2/episodes/episode_000000/4.16.mp4 \
  --sequence-name 4.16 \
  --remote-project-root /data2/hrd/Dyn-HaMR \
  --remote-test-root /data2/hrd/Dyn-HaMR/test \
  --remote-video-dir /data2/hrd/Dyn-HaMR/test/videos \
  --local-output-dir data/hamer_outputs
```

说明：脚本会按顺序执行上传、远程 conda 激活与推理、结果回传。默认使用 SSH 别名 `yyl`。

批量处理整个 `episodes/`（同时处理 `d435/rgb.mp4` 和 `top/rgb.mp4`）：

```bash
记得设置全局变量名
RUN_NAME = 5.5 或 export RUN_NAME = 5.5

python pipelines/hamer/8_dynhamr_remote_sync.py \
  --episodes-root data/raw/${RUN_NAME} \
  --remote-project-root /data2/hrd/Dyn-HaMR \
  --remote-test-root /data2/hrd/Dyn-HaMR/test \
  --remote-video-dir /data2/hrd/Dyn-HaMR/test/videos \
  --local-output-dir data/hamer_outputs/${RUN_NAME} \
  --gpu-id 3 \
  --dynhamr-fps 30 \
  --skip-existing

清理下残留文件
ssh yyl 
rm -rf /data2/hrd/Dyn-HaMR/test/images/5.5_d435_chunk-000_file-000 \
       /data2/hrd/Dyn-HaMR/test/dynhamr/cameras/5.5_d435_chunk-000_file-000 \
       /data2/hrd/Dyn-HaMR/test/dynhamr/track_preds/5.5_d435_chunk-000_file-000 \
       /data2/hrd/Dyn-HaMR/test/dynhamr/shot_idcs/5.5_d435_chunk-000_file-000.json \
       /data2/hrd/Dyn-HaMR/test/dynhamr/hamer_out/5.5_d435_chunk-000_file-000 \
       /data2/hrd/Dyn-HaMR/test/videos/5.5_d435_chunk-000_file-000.mp4 
```


手动登录服务器：

```bash
ssh yyl
```

可选参数：
- `--only-d435` 只处理 `d435/rgb.mp4`
- `--only-top` 只处理 `top/rgb.mp4`
- `--dry-run` 仅打印将执行的命令
- `--skip-existing` 本地已有结果时跳过

### 4) 轨迹提取

```bash
PYTHONPATH=. python pipelines/extract/3_extract_traj.py \
  --input-pkl data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
  --episode-dir data/raw/${RUN_NAME}/depth/d435/chunk-000/file-000 \
  --output-csv data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
  --fps 30 \
  --hand right \
  --patch-radius 0 \
  --max-missing-gap 8 \


```

```bash
--patch-radius 含义是取深度时的邻域半径（单位像素）：

0：只取中心像素
1：取 3x3 邻域中值
2：取 5x5 邻域中值
3：取 7x7 邻域中值
```
`--transform-json` 可省略，默认读取 `calibration/outputs/cam2base_latest.json`。
注意：新版提取脚本强制使用深度信息，`--episode-dir` 下必须存在 `depth.npz` 和 `intrinsics.json`。

### 5) 轨迹可视化验收

```bash
PYTHONPATH=. python pipelines/visualize/4_visualize_traj.py \
  --input-pkl data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
  --episode-dir data/raw/${RUN_NAME}/depth/d435/chunk-000/file-000 
```

`如果要导出视频，加上这个参数 --export-only`
```bash
PYTHONPATH=. python pipelines/visualize/4_visualize_traj.py \
  --input-pkl data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
  --episode-dir data/raw/${RUN_NAME}/depth/d435/chunk-000/file-000 \
  --export-only
```

`--transform-json` 可省略，默认读取 `calibration/outputs/cam2base_latest.json`。

### 6) 结构化轨迹平滑（P1: temporal + bone + reprojection）

```bash
python pipelines/extract/3b_smooth_traj_structured.py \
  --input-csv data/replay_csv/<name>.csv \
  --episode-dir data/raw/${RUN_NAME}/depth/d435/chunk-000/file-000 \
  --output-csv data/replay_csv/<name>_smoothed.csv
```

### 7) 导出 VLA 训练表（Relative + Chunk）

```bash
python pipelines/export/export_vla_dataset.py \
  --input-csv data/replay_csv/<name>_smoothed.csv \
  --output-csv data/vla_ready/<name>_vla.npz \
  --output-format npz \
  --chunk-size 10 \
  --keep-hand-found-only
```

### 8) 校验 VLA 导出质量

```bash
python pipelines/export/validate_vla_dataset.py \
  --input-path data/vla_ready/<name>_vla.npz
```

### 9) 终端抓取先验（GraspNet 桥接）

```bash
python pipelines/grasp/5_infer_grasp_prior.py \
  --data-dir third-party/graspnet-baseline/doc/my_grasp_data \
  --checkpoint-path third-party/graspnet-baseline/logs/log_rs/checkpoint-rs.tar \
  --output-json artifacts/grasp_prior/top_grasp.json
```

输出文件包含 `top_grasp.translation_cam_xyz / translation_base_xyz`，可在 pinch 低置信时作为终端抓取锚点。

### 10) 抓取窗口检测（pinch 低置信触发）

```bash
python pipelines/grasp/6_detect_grasp_window.py \
  --input-csv data/replay_csv/<name>_smoothed.csv \
  --output-json artifacts/grasp_prior/grasp_window.json
```

输出 `grasp_window.json`（窗口起止帧）和同目录 `*.scored.csv`（逐帧打分）。

### 11) 末段抓取锚点融合（生成 fused 轨迹）

```bash
python pipelines/grasp/7_fuse_grasp_anchor.py \
  --input-csv data/replay_csv/<name>_smoothed.csv \
  --grasp-prior-json artifacts/grasp_prior/top_grasp.json \
  --grasp-window-json artifacts/grasp_prior/grasp_window.json \
  --output-csv data/replay_csv/<name>_fused.csv
```

建议后续 VLA 导出优先使用 `*_fused.csv`。

### 12) 运行控制

```bash
python run_pipeline.py --mode keyboard
  #pitch上负下正，roll左负右正
  #P 显示你当前实际在转的 pitch（对应内部 roll 变量的偏移）
  #R 显示实际 roll（对应内部 pitch 变量）

python run_pipeline.py --mode mujoco

python run_pipeline.py --mode vision

python run_pipeline.py --mode replay \
  --csv-path /home/rita/HandGuiding_test/project/data/replay_csv/5.5/5.5_d435_chunk-000_file-000_abs.csv \
  --port /dev/ttyACM1 \
  --fps 30 \
  --target-offset-x 0.03 \
  --target-offset-y 0.0 \
  --target-offset-z 0.0
-0.0203
```

`--mode replay` 现已支持统一轨迹日志导出（运行时按提示输入路径，默认输出到 `data/vla_ready/replay_logs/replay_latest.jsonl`）。

说明：
- 标定流程（`calibration/1_dual_camera_robot_click_calibration.py`）已经内置键盘控臂，不需要再并行启动 `python run_pipeline.py --mode keyboard`。
- 不要同时启动两个机械臂控制进程；同一串口（如 `/dev/ttyACM0`）会冲突，导致其中一个进程无法控制机械臂。

兼容入口（仍可用）：

```bash
python 2_control_ee_with_pinocchio_so100_real.py
python 3_vision_grasp.py
python 4_replay_csv_so100.py
```

## Notes

- 推荐把 `data/` 与 `artifacts/` 作为本地目录，不直接提交大型文件到仓库。
- 详细框架说明见 `FRAMEWORK.md`，任务拆解见 `task.md`。

## Change Log

### 2026-04-27

- `pipelines/extract/3_extract_traj.py`
  - 接入真实时间戳对齐：优先使用 `timestamps.npy`，其次 `frame_meta.jsonl`，最后才回退 `frame_index/fps`。
  - 新增 `timestamp_s` 列；`t` 改为相对真实时间。
  - 增加日志输出 `Timestamp source`，便于检查对齐来源。
- `calibration/2_handeye_calibration.py`
  - 从“硬编码点对”升级为“读点文件求解”（支持 CSV/JSON）。
  - 默认读 `calibration/points/handeye_points.csv`。
  - 自动输出到 `calibration/outputs/cam2base_latest.json`，并同步兼容副本 `record/json/cam2base.json`。
  - 输出 RMSE/最大误差等标定统计信息。
- `calibration/points/handeye_points.csv`
  - 新增初始点对文件，作为标定输入模板。
- `pipelines/export/export_vla_dataset.py`
  - 新增 VLA 训练数据导出脚本。
  - 将轨迹表转换为 `obs_* + action_rel_* + action_rel_step{k}_*` 格式。
  - 支持 `chunk-size`、`hand_found` 过滤、时间戳自动回退策略。
  - 支持直接导出 `npz`（不再需要后续格式转换）。
