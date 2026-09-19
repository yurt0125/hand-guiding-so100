# Hand Guiding SO100/SO101 Pipeline

SO100/101 机械臂手部示教项目主流程：
标定 -> 录制 -> Dyn-HaMR -> 轨迹提取 -> 可视化验收 -> replay 导出 action -> 转 LeRobot 数据集 -> 训练 -> 推理。

###  运行控制

```bash
python run_pipeline.py --mode keyboard --port /dev/ttyACM1
  # x: 左正右负；y: 前负后正；z: 上正下负
  # pitch: 上负下正；roll: 左负右正

python run_pipeline.py --mode mujoco

python run_pipeline.py --mode vision

python run_pipeline.py --mode replay \
  --csv-path /home/robot/hand-guiding-so100/data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
  --port /dev/ttyACM1 \
  --fps 30 \
  --target-offset-x 0.02 \
  --target-offset-y 0.0 \
  --target-offset-z -0.03
```

标定
lerobot-calibrate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=single_arm



## 0) 运行名

```bash
# export RUN_NAME=export RUN_NAME=5.8_act_test2 #25条人手香蕉 老格式csv不适用了
# export RUN_NAME=export RUN_NAME=5.15_test   #螺丝刀数据集
# export RUN_NAME=export RUN_NAME=5.24_30bana  #补充30条人手香蕉
# export RUN_NAME=export RUN_NAME=5.24_20bana  #补充20条人手香蕉
# export RUN_NAME=export RUN_NAME=5.24_merged50_bana
# export RUN_NAME=export RUN_NAME=7.24_bana1
# export RUN_NAME=export RUN_NAME=7.24_table1



export RUN_NAME=export RUN_NAME=8.4_banana1
export RUN_NAME=export RUN_NAME=8.4_banana2
export RUN_NAME=export RUN_NAME=8.4_banana3
export RUN_NAME=export RUN_NAME=8.4_banana4
export RUN_NAME=export RUN_NAME=8.4_banana56
export RUN_NAME=export RUN_NAME=8.4_banana78
export RUN_NAME=export RUN_NAME=8.4_banana910



export RUN_NAME=export RUN_NAME=8.5_blocks1
export RUN_NAME=export RUN_NAME=8.5_blocks2
export RUN_NAME=export RUN_NAME=8.5_blocks34
export RUN_NAME=export RUN_NAME=8.5_blocks56
export RUN_NAME=export RUN_NAME=8.5_blocks78
export RUN_NAME=export RUN_NAME=8.5_blocks910



export RUN_NAME=export RUN_NAME=8.7_stack1
export RUN_NAME=export RUN_NAME=8.7_stack2
export RUN_NAME=export RUN_NAME=8.7_stack34
export RUN_NAME=export RUN_NAME=8.7_stack56
export RUN_NAME=export RUN_NAME=8.7_stack78
export RUN_NAME=export RUN_NAME=8.7_stack910

export RUN_NAME=export RUN_NAME=8.17_push_test
export RUN_NAME=export RUN_NAME=8.17_push1
export RUN_NAME=export RUN_NAME=8.17_push2
export RUN_NAME=export RUN_NAME=8.17_push3
export RUN_NAME=export RUN_NAME=8.17_push4
export RUN_NAME=export RUN_NAME=8.17_push5
export RUN_NAME=export RUN_NAME=8.17_push6
export RUN_NAME=export RUN_NAME=8.17_push7
export RUN_NAME=export RUN_NAME=8.17_push8
export RUN_NAME=export RUN_NAME=8.17_push9
export RUN_NAME=export RUN_NAME=8.17_push10

export RUN_NAME=export RUN_NAME=8.11_flip_test
export RUN_NAME=export RUN_NAME=8.11_flip1
export RUN_NAME=export RUN_NAME=8.11_flip2

export RUN_NAME=export RUN_NAME=8.12_button_test
export RUN_NAME=export RUN_NAME=8.12_button12
export RUN_NAME=export RUN_NAME=8.12_button34
export RUN_NAME=export RUN_NAME=8.12_button56
export RUN_NAME=export RUN_NAME=8.12_button78
export RUN_NAME=export RUN_NAME=8.12_button910

export RUN_NAME=export RUN_NAME=8.15_hanging_test
export RUN_NAME=export RUN_NAME=8.15_hanging1
export RUN_NAME=export RUN_NAME=8.15_hanging2
export RUN_NAME=export RUN_NAME=8.15_hanging3
export RUN_NAME=export RUN_NAME=8.15_hanging4
export RUN_NAME=export RUN_NAME=8.15_hanging5

export RUN_NAME=export RUN_NAME=8.15_hanging67-12

export RUN_NAME=export RUN_NAME=8.15_hanging67-3
export RUN_NAME=export RUN_NAME=8.15_hanging67-4
export RUN_NAME=export RUN_NAME=8.15_hanging67-5
export RUN_NAME=export RUN_NAME=8.15_hanging67-6

export RUN_NAME=export RUN_NAME=8.15_hanging67-7
export RUN_NAME=export RUN_NAME=8.15_hanging67-8
export RUN_NAME=export RUN_NAME=8.15_hanging67-9
export RUN_NAME=export RUN_NAME=8.15_hanging67-10

export RUN_NAME=export RUN_NAME=8.15_hanging89-1
export RUN_NAME=export RUN_NAME=8.15_hanging89-2
export RUN_NAME=export RUN_NAME=8.15_hanging89-3
export RUN_NAME=export RUN_NAME=8.15_hanging89-4

export RUN_NAME=export RUN_NAME=8.15_hanging89-5
export RUN_NAME=export RUN_NAME=8.15_hanging89-6
export RUN_NAME=export RUN_NAME=8.15_hanging89-7
export RUN_NAME=export RUN_NAME=8.15_hanging89-8

export RUN_NAME=export RUN_NAME=8.15_hanging89-9
export RUN_NAME=export RUN_NAME=8.15_hanging89-10
export RUN_NAME=export RUN_NAME=8.15_hanging10-1
export RUN_NAME=export RUN_NAME=8.15_hanging10-2

export RUN_NAME=export RUN_NAME=8.15_hanging10-3
export RUN_NAME=export RUN_NAME=8.15_hanging10-4
export RUN_NAME=export RUN_NAME=8.15_hanging10-5

export RUN_NAME=export RUN_NAME=9.17_banana-1




```

## 1) 手眼标定（可跳过）

```bash
python calibration/1_dual_camera_robot_click_calibration.py \
  --robot-port /dev/ttyACM1 \
  --rs-serial 216322074780 \
  --fps 15 \
  --control-hz 30 \
  --min-pairs 30

# 如果机械臂仍然卡顿，加上 --profile-control 查看 IK/串口写入耗时。

lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=single_arm \
  --robot.cameras='{
    "d435": {
      "type": "intelrealsense",
      "serial_number_or_name": "216322074780",
      "width": 640,
      "height": 480,
      "fps": 30,
      "use_depth": true
    }
  }' \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader \
  --display_data=true



python calibration/2_click_to_base_verify.py \
  --transform-json calibration/outputs/cam2base_latest.json \
  --patch-radius 0
```

默认标定文件：
- `calibration/outputs/cam2base_latest.json`

## 2) 录制原始数据（D435）
说明：该录制脚本写入的数据集元信息中 `robot_type=hand`。 录制不宜太长，容易爆掉

```bash
export PYTHONPATH="third-party/lerobot/src:/home/robot/hand-guiding-so100"
export MKL_THREADING_LAYER=GNU
PYTHONPATH=third-party/lerobot/src:. python pipelines/record/3_rawdata_record.py \
  --camera-only \
  --export-depth-npz \
  --robot.cameras="{
    'd435': {'type': 'intelrealsense', 'serial_number_or_name': '216322074780', 'width': 640, 'height': 480, 'fps': 30, 'use_depth': true}
  }" \
  --dataset.root=/home/robot/hand-guiding-so100/data/raw/${RUN_NAME} \
  --dataset.repo_id=RITAHuang/${RUN_NAME} \
  --display_data=true \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=30 \
  --dataset.push_to_hub=false \
  --dataset.single_task="d435 only collection" \
  --dataset.fps=30
```
    --resume=true的逻辑还没整定，后续看看怎么处理更加优雅

录制结束后，如果 `depth/d435/episodes/` 下已经有 `episode-*.npz`，执行一次深度合成：

```bash
PYTHONPATH=third-party/lerobot/src:. python pipelines/record/3_merge_depth_episodes.py \
  --dataset-root /home/robot/hand-guiding-so100/data/raw/${RUN_NAME}   --camera-key d435
```

可选：加 `--uncompressed` 可更快写出（文件更大）。

```bash
--camera-key d435 是告诉脚本去处理哪一路相机的深度目录，也就是：

data/raw/${RUN_NAME}/depth/d435/episodes/episode-*.npz
如果以后有别的相机 key（比如 wrist），就改成 --camera-key wrist。
```

## 3) Dyn-HaMR 远程处理

```bash
python pipelines/hamer/4_dynhamr_remote_process.py \
  --episodes-root data/raw/${RUN_NAME} \
  --remote-project-root /data2/hrd/Dyn-HaMR \
  --remote-test-root /data2/hrd/Dyn-HaMR/test \
  --remote-video-dir /data2/hrd/Dyn-HaMR/test/videos \
  --local-output-dir data/hamer_outputs/${RUN_NAME} \
  --dynhamr-fps 30 \
  --gpu-id 0 
```


<!-- 合并多份 HaMeR 输出（pkl + render 视频，命名与既有流程一致）：
```bash
python /home/robot/hand-guiding-so100/pipelines/hamer/6_merge_hamer_packages.py \
  --hamer-input-dirs \
  /home/robot/hand-guiding-so100/data/hamer_outputs/5.24_20bana \
  /home/robot/hand-guiding-so100/data/hamer_outputs/5.24_30bana \
  --output-dir /home/robot/hand-guiding-so100/data/hamer_outputs \
  --merged-name 5.24_merged_bana
```
如需显式指定 raw 输入/输出（不再按名称自动推断）：
```bash
export RUN_NAME=export RUN_NAME=5.24_merged50_bana

python /home/robot/hand-guiding-so100/pipelines/hamer/6_merge_hamer_packages.py \
  --hamer-input-dirs \
  /home/robot/hand-guiding-so100/data/hamer_outputs/5.24_20bana \
  /home/robot/hand-guiding-so100/data/hamer_outputs/5.24_30bana \
  --output-dir /home/robot/hand-guiding-so100/data/hamer_outputs \
  --raw-input-dirs \
  /home/robot/hand-guiding-so100/data/raw/5.24_20bana \
  /home/robot/hand-guiding-so100/data/raw/5.24_30bana \
  --raw-output-dir /home/robot/hand-guiding-so100/data/raw \
  --merged-name 5.24_merged50_bana
```
说明：输出将为
- `data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl`
- `data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/results/render_all_500.0.mp4` -->


banana_config
  --pinch-close-m 0.035 \
  --pinch-open-m 0.15

stacks_config
  --pinch-close-m 0.035 \
  --pinch-open-m 0.125

hanging_config
  --pinch-close-m 0.03 \
  --pinch-open-m 0.065

## 4) 轨迹提取（abs.csv）

```bash
PYTHONPATH=. python pipelines/extract/3_extract_traj.py \
  --input-pkl data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
  --episode-dir data/raw/${RUN_NAME}/depth/d435 \
  --output-csv data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
  --fps 30 \
  --hand right \
  --patch-radius 0 \
  --max-missing-gap 8 \
  --pinch-close-m 0.035 \
  --pinch-open-m 0.15
```

## 5) 轨迹可视化验收（Rerun）

步进模式：
```bash
# PYTHONPATH=. python pipelines/visualize/4_visualize_traj_rerun.py \
#   --input-pkl data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
#   --csv-path data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
#   --episode-dir data/raw/${RUN_NAME}/depth/d435 \
#   --rgb-path data/raw/${RUN_NAME}/videos/observation.images.d435/chunk-000/file-000.mp4 \
#   --step-through \
#   --start-frame 0
```

自动播放：
```bash
PYTHONPATH=. python pipelines/visualize/4_visualize_traj_rerun.py \
  --input-pkl data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
  --csv-path data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
  --episode-dir data/raw/${RUN_NAME}/depth/d435 \
  --rgb-path data/raw/${RUN_NAME}/videos/observation.images.d435/chunk-000/file-000.mp4
```

## 6) replay + 导出真实下发 action（sent_action.csv）

只导出 action（不连实机）：
```bash
# python run_pipeline.py --mode replay \
#   --csv-path /home/robot/hand-guiding-so100/data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
#   --fps 30 \
#   --target-offset-x 0.03 \
#   --target-offset-y 0.0 \
#   --target-offset-z -0.03 \
#   --export-action-csv /home/robot/hand-guiding-so100/data/replay_csv/${RUN_NAME}/${RUN_NAME}_sent_action.csv
```

banana_config
  --target-offset-x 0.015 \
  --target-offset-y 0.005 \
  --target-offset-z -0.03 \

blocks_config
  --target-offset-x 0.015 \
  --target-offset-y -0.01 \
  --target-offset-z -0.03 \

stack_config
  --target-offset-x 0.015 \
  --target-offset-y -0.005 \
  --target-offset-z -0.01 \

push_config
  --target-offset-x -0.002 \
  --target-offset-y 0.00 \
  --target-offset-z -0.01 \

hanging_config
  --target-offset-x -0.002 \
  --target-offset-y -0.006 \
  --target-offset-z -0.004 \

  --target-offset-x 0.002 \
  --target-offset-y -0.005 \
  --target-offset-z -0.00 \

连实机 replay：
```bash
PYTHONPATH=third-party/lerobot/src:. python run_pipeline.py --mode replay \
  --csv-path data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
  --port /dev/ttyACM1 \
  --fps 30 \
  --target-offset-x 0.015 \
  --target-offset-y 0.005 \
  --target-offset-z -0.03 \
  --episode-reset-time-s 8 \
  --profile-control
```

replay的同时会导出机械臂的视频：
对齐逻辑为：每下发 1 条 action，就采集并写入 1 帧视频
```bash
python run_pipeline.py --mode replay \
  --csv-path /home/robot/hand-guiding-so100/data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
  --port /dev/ttyACM1 \
  --fps 30 \
  --target-offset-x 0.015 \
  --target-offset-y 0.005 \
  --target-offset-z -0.03 \
  --episode-reset-time-s 12 \
  --export-action-csv /home/robot/hand-guiding-so100/data/replay_csv/${RUN_NAME}/${RUN_NAME}_sent_action.csv \
  --record-video-path /home/robot/hand-guiding-so100/data/replay_videos/${RUN_NAME}_robot.mp4 \
  --record-rs-serial 216322074780 \
  --record-video-width 640 \
  --record-video-height 480 \
  --record-video-fps 30
```


## 7) 生成 ACT 训练数据集（sent_action 对齐版）

```bash
PYTHONPATH=third-party/lerobot/src:. python pipelines/train/prepare_act_dataset_from_csv.py \
  --dataset-root /home/robot/hand-guiding-so100/data/raw/${RUN_NAME} \
  --output-root /home/robot/hand-guiding-so100/data/act/${RUN_NAME}_from_sent_action \
  --repo-id ${RUN_NAME}_from_sent_action \
  --action-csv-path /home/robot/hand-guiding-so100/data/replay_csv/${RUN_NAME}/${RUN_NAME}_sent_action.csv \
  --pkl-path /home/robot/hand-guiding-so100/data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
  --hand-video-path /home/robot/hand-guiding-so100/data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/results/render_all_500.0.mp4 \
  --robot-video-path /home/robot/hand-guiding-so100/data/replay_videos/${RUN_NAME}/${RUN_NAME}_robot.mp4 \
  --video-backend pyav
```

输出目录示例：
- `data/act/${RUN_NAME}_from_sent_action`

## 8) LeRobot 官方 replay 验证

```bash
lerobot-replay \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=single_arm \
  --robot.use_degrees=true \
  --dataset.repo_id=${RUN_NAME}_from_sent_action \
  --dataset.root=/home/robot/hand-guiding-so100/data/act/${RUN_NAME}_from_sent_action \
  --dataset.episode=0
```

## 9) ACT 训练

### 9.1 本地（rita）

```bash
RUN_NAME=5.8_act_test2_from_sent_action

lerobot-train \
  --dataset.repo_id=RITAHuang/${RUN_NAME} \
  --dataset.root=/home/robot/hand-guiding-so100/data/act/${RUN_NAME} \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir=/home/robot/hand-guiding-so100/artifacts/train_runs/act/${RUN_NAME} \
  --job_name=${RUN_NAME}_act_train \
  --policy.device=cuda \
  --batch_size=128 \
  --wandb.enable=false \
  --policy.repo_id=RITAHuang/${RUN_NAME}_policy \
  --policy.push_to_hub=true \
  --steps=40000 \
  --save_freq=8000
```

### 9.2 服务器离线训练（omnisky）

先把数据集放到默认目录：
- `/home/robot/.cache/huggingface/lerobot/RITAHuang/${RUN_NAME}`

```bash
RUN_NAME=5.8_act_test2_from_sent_action

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
lerobot-train \
  --dataset.repo_id=${RUN_NAME} \
  --dataset.root=/home/robot/.cache/huggingface/lerobot/RITAHuang/${RUN_NAME} \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --batch_size=64 \
  --job_name=${RUN_NAME}_act_train \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=40000 \
  --save_freq=8000
```

## 10) ACT 推理录制（policy rollout）

```bash
RUN_NAME=5.8_act_test2_from_sent_action

lerobot-record \
  --robot.type=so100_follower --robot.port=/dev/ttyACM1 --robot.id=leader \
  --teleop.type=so100_leader --teleop.port=/dev/ttyACM0 --teleop.id=3204 \
  --robot.disable_torque_on_disconnect=true \
  --robot.cameras="{'d435': {'type': 'intelrealsense', 'serial_number_or_name': '216322074780', 'width': 640, 'height': 480, 'fps': 30}}" \
  --display_data=true \
  --dataset.single_task="Grab the banana" \
  --policy.path=/home/robot/hand-guiding-so100/artifacts/train_runs/act/${RUN_NAME}/checkpoints/016000/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=RITAHuang/eval_so100_${RUN_NAME} \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=40 \
  --dataset.num_episodes=10
```
rm -rf /home/robot/.cache/huggingface/lerobot/RITAHuang/eval_so100_${RUN_NAME}

## 10.1) YOLO-seg + RealSense(+ICP) 快速测试

```bash
python pipelines/vision/test_yolo_seg_realsense.py \
  --model /home/robot/hand-guiding-so100/models/yolov8n-seg.pt \
  --serial 216322074780 \
  --width 640 --height 480 --fps 30 \
  --target-class banana \
  --conf 0.35 \
  --device 0
```

## 10.2) Rerun 可视化中启用 YOLO-seg（叠加到轨迹画面）

```bash
PYTHONPATH=. python pipelines/visualize/4_visualize_traj_rerun.py \
  --input-pkl data/hamer_outputs/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000/${RUN_NAME}_d435_chunk-000_file-000.pkl \
  --csv-path data/replay_csv/${RUN_NAME}/${RUN_NAME}_d435_chunk-000_file-000_abs.csv \
  --episode-dir data/raw/${RUN_NAME}/depth/d435 \
  --rgb-path data/raw/${RUN_NAME}/videos/observation.images.d435/chunk-000/file-000.mp4 \
  --enable-yolo-seg \
  --yolo-model /home/robot/hand-guiding-so100/models/yolov8n-seg.pt \
  --yolo-target-class banana \
  --yolo-conf 0.35 \
  --yolo-device 0
```

## 11) 常见问题

1. `HFValidationError: Repo id must be ... /path/...`
- 原因：`--policy.path` 指向的目录不是完整 pretrained_model 目录。
- 处理：优先用 `checkpoints/last/pretrained_model`。

2. `FileNotFoundError: .../meta/info.json`
- 原因：`dataset.root` 层级写错或数据集没同步。
- 处理：先 `ls <dataset_root>/meta/info.json` 验证。

3. 服务器无法登录 HuggingFace
- 现象：`ProxyError` 或 `ConnectTimeout`。
- 处理：离线训练（`HF_HUB_OFFLINE=1` + `--policy.push_to_hub=false`），训练后本地上传。

## 12) 暂不使用

以下流程暂不在主链路：
- `3b_smooth_traj_structured.py`
- `export_vla_dataset.py`
- `validate_vla_dataset.py`
- `5_infer_grasp_prior.py`



```bash
2026.5.15  修改与思考

1、末端控制点位置的选择影响手腕旋转效果	     修改xml文件，将末端控制点从Jaw移动到JawOffset（位于正中心）

2、标定文件中点击点的位置不是实际IK出来末端点的位置	     利用刚体变换把坐标点定在JAW上，通过JAW点的IK坐标与像素坐标对应来实现标定

3、魔改官方数据采集代码中--resume无法正常恢复深度的录制	    修改深度录制逻辑

4、在上次数据录制中发现，hamr会偶然将右手检测成左手导致数据不可用	   原逻辑是将通过标志位检测不合格数据对应删除，会造成数据集的浪费。现将hamr中run.py的逻辑进行修改，强制指定右手

5、Rerun轨迹可视化中可以把qpos曲线显示出来,便于debug	

6、提取轨迹时指尖距离和夹爪开合度之间靠肉眼线性映射	可行，但不是最优

7、replay过程的目标点必须给少量偏置修正，如：  --target-offset-x 0.02 \ --target-offset-y 0.0  --target-offset-z -0.03，	人手和夹爪实际体积有差，得想更好的方法映射，目前少量修正的方法可行但不是最优

8、回放时，机械臂起点和可运行第一帧之间有一小段距离，会突然移动过去，不够优雅	在他们直接直接插入几个点（图像也需对应）
```

## 官方遥操作命令（SO101）

```bash
# lerobot-teleoperate \
#   --robot.type=so100_follower \
#   --robot.port=/dev/ttyACM1 \
#   --robot.id=single_arm \
#   --teleop.type=so100_leader \
#   --teleop.port=/dev/ttyACM0 \
#   --teleop.id=leader

lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=single_arm \
  --robot.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/robots/so100_follower \
  --robot.disable_torque_on_disconnect=true \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader \
  --teleop.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader


export RUN_NAME=smolvla_banana_910
export RUN_NAME=smolvla_blocks_910
export RUN_NAME=smolvla_push_910
export RUN_NAME=smolvla_stacks_910
export RUN_NAME=smolvla_hanging_910
export RUN_NAME=smolvla_pull_910




lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=single_arm \
  --robot.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/robots/so100_follower \
  --robot.cameras="{d435: {type: intelrealsense, serial_number_or_name: 216322074780, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader \
  --teleop.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader \
  --dataset.repo_id=RITAHuang/${RUN_NAME} \
  --dataset.root=/home/robot/.cache/huggingface/lerobot/RITAHuang/${RUN_NAME} \
  --dataset.single_task="hanging" \
  --dataset.fps=30 \
  --dataset.episode_time_s=40 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=true


  --resume=true

```

## yyl2：合并香蕉数据集上传与官方 SmolVLA 训练（GPU 7）

本地将 50 条合并数据集上传到 `yyl2`：

```bash
rsync -a --progress \
  /home/robot/.cache/huggingface/lerobot/RITAHuang/smolvla_banana_0_49_merged/ \
  yyl2:/home/yyl/.cache/huggingface/lerobot/RITAHuang/smolvla_banana_0_49_merged/
```

服务器使用官方 LeRobot 0.4.3 SmolVLA 微调入口，并固定使用物理 GPU 7：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 \
/home/yyl/anaconda3/envs/lerobot/bin/lerobot-train \
  --dataset.repo_id=RITAHuang/smolvla_banana_0_49_merged \
  --dataset.root=/home/yyl/.cache/huggingface/lerobot/RITAHuang/smolvla_banana_0_49_merged \
  --dataset.video_backend=pyav \
  --policy.path=/data1/hrd/models/smolvla_base \
  --policy.vlm_model_name=/data1/hrd/models/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null \
  --policy.output_features=null \
  --output_dir=/home/yyl/lerobot_outputs/smolvla/smolvla_banana_0_49_merged_run1 \
  --job_name=smolvla_banana_0_49_merged_train \
  --policy.device=cuda \
  --batch_size=64 \
  --num_workers=12 \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=10000 \
  --save_freq=5000
```

## 本地：训练完成的 SmolVLA 策略推理（policy rollout）

模型下载至本机 `training_outputs/` 后，使用官方 `lerobot-record` 进行策略推理。以下以 banana 模型为例；主臂仅用于 episode 间的安全复位，正式 rollout 仍由策略控制。

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=single_arm \
  --robot.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/robots/so100_follower \
  --robot.cameras="{'d435': {'type': 'intelrealsense', 'serial_number_or_name': '216322074780', 'width': 640, 'height': 480, 'fps': 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader \
  --teleop.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader \
  --policy.path=/home/robot/hand-guiding-so100/training_outputs/smolvla_banana_0_49_merged_run1/checkpoints/010000/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=RITAHuang/eval_smolvla_banana_round6 \
  --dataset.root=/home/robot/.cache/huggingface/lerobot/RITAHuang/eval_smolvla_banana_round6 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the banana" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=30 \
  --dataset.num_episodes=10 \
  --display_data=true \
  --play_sounds=true
```


- banana checkpoint 已配置为使用本机 `/home/robot/hand-guiding-so100/models/SmolVLM2-500M-Video-Instruct`；该目录必须保留。
- 对其他任务，替换 `--policy.path` 内的任务目录和 `--dataset.single_task`；最终模型均在 `checkpoints/010000/pretrained_model`。

### 六个任务的统一 rollout 命令

先执行一次下方函数定义；随后每次只执行一条任务调用。主臂只在 episode 间的 reset 环节控制从臂，正式 rollout 由策略控制。`round` 名可改为未使用的编号，避免混入已有评估数据。

```bash
run_smolvla_rollout() {
  local task="$1" run="$2" task_text="$3" eval_round="$4" checkpoint="${5:-010000}"
  lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=single_arm \
    --robot.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/robots/so100_follower \
    --robot.disable_torque_on_disconnect=true \
    --robot.cameras="{'d435': {'type': 'intelrealsense', 'serial_number_or_name': '216322074780', 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader \
    --teleop.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader \
    --policy.path="/home/robot/hand-guiding-so100/training_outputs/smolvla_${task}_0_49_merged_${run}/checkpoints/${checkpoint}/pretrained_model" \
    --policy.device=cuda \
    --dataset.repo_id="RITAHuang/eval_smolvla_${task}_${eval_round}" \
    --dataset.root="/home/robot/.cache/huggingface/lerobot/RITAHuang/eval_smolvla_${task}_${eval_round}" \
    --dataset.push_to_hub=false \
    --dataset.single_task="$task_text" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.num_episodes=10 \
    --display_data=true \
    --play_sounds=true
}

# 每次仅取消一条命令开头的 #：

# run1：保留旧模型及其训练时使用的旧文本
run_smolvla_rollout banana   run1  "Pick up the banana"  round8
run_smolvla_rollout blocks   run1  "Pick up the banana"  round2
run_smolvla_rollout hanging  run1  "hanging"             round1
run_smolvla_rollout stacks   run1  "stacks"              round2
run_smolvla_rollout pull     run1  "hanging"             round1
run_smolvla_rollout push     run1  "Pick up the banana"  round3

# run2：对应模型训练完成后再执行
run_smolvla_rollout blocks   run2  "Put the green block in the basket"      round1
run_smolvla_rollout push     run2  "Push the green block to the target area" round1
run_smolvla_rollout pull     run2  "Pull the storage box"                    round1
run_smolvla_rollout stacks   run2  "Stack the blocks"                        round1
run_smolvla_rollout hanging  run2  "Hang the tape"                           round1

# 测试 run1 的 5,000-step checkpoint（使用它训练时的原任务文本）：
run_smolvla_rollout stacks  run1  "stacks"              round3 005000
run_smolvla_rollout push    run1  "Pick up the banana"  round3 005000
```

<!-- 
## yyl2：合并 blocks 数据集上传与官方 SmolVLA 训练（GPU 4）

```bash
# 本地合并五个 10-episode 批次（保留原始批次）
lerobot-edit-dataset \
  --repo_id=RITAHuang/smolvla_blocks_0_49_merged \
  --operation.type=merge \
  --operation.repo_ids="['RITAHuang/smolvla_blocks_12', 'RITAHuang/smolvla_blocks_34', 'RITAHuang/smolvla_blocks_56', 'RITAHuang/smolvla_blocks_78', 'RITAHuang/smolvla_blocks_910']" \
  --push_to_hub=false

# 上传至 yyl2
rsync -a --progress \
  /home/robot/.cache/huggingface/lerobot/RITAHuang/smolvla_blocks_0_49_merged/ \
  yyl2:/home/yyl/.cache/huggingface/lerobot/RITAHuang/smolvla_blocks_0_49_merged/

# yyl2：使用空闲的物理 GPU 4 进行离线 SmolVLA 训练
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 \
/home/yyl/anaconda3/envs/lerobot/bin/lerobot-train \
  --policy.path=/data1/hrd/models/smolvla_base \
  --policy.vlm_model_name=/data1/hrd/models/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset.repo_id=RITAHuang/smolvla_blocks_0_49_merged \
  --dataset.root=/home/yyl/.cache/huggingface/lerobot/RITAHuang/smolvla_blocks_0_49_merged \
  --output_dir=/home/yyl/lerobot_outputs/smolvla/smolvla_blocks_0_49_merged_run1 \
  --job_name=smolvla_blocks_0_49_merged_train \
  --policy.device=cuda \
  --batch_size=64 \
  --num_workers=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=10000 \
  --save_freq=5000
``` -->

lerobot-replay \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=single_arm \
  --robot.calibration_dir=/home/robot/.cache/huggingface/lerobot/calibration/robots/so100_follower \
  --robot.disable_torque_on_disconnect=true \
  --dataset.repo_id=RITAHuang/smolvla_push_0_49_merged \
  --dataset.root=/home/robot/.cache/huggingface/lerobot/RITAHuang/smolvla_push_0_49_merged \
  --dataset.episode=3 \
  --dataset.fps=30 \
  --play_sounds=false