# SO100 手部示教项目任务看板（重构后）

## 1. 当前代码结构（已完成重整）

```text
project/
  app/
    keyboard_mirror_app.py          # 主控：真机+MuJoCo镜像（保留细节）
    keyboard_app_refactored.py      # 轻量键盘示教入口
    vision_app_refactored.py        # 视觉点击目标入口
    replay_app_refactored.py        # CSV 回放入口
  common/
    common_robot.py                 # 运动学、硬件接口、关节映射
    common_control.py               # Smoothing / Guard / JointFilter / Runner
    config.py                       # 工作空间与控制参数集中配置
  calibration/
    1_d435_click.py
    2_handeye_calibration.py
    outputs/cam2base_latest.json
  pipelines/
    record/1_d435_episode_record.py
    extract/3_extract_traj.py
    visualize/4_visualize_traj.py
  data/
    raw_episodes/
    hamer_outputs/
    replay_csv/
    trajectories/
    vla_ready/
  artifacts/
    previews/
    debug_videos/
  legacy/
    1_control_ee_with_pinocchio_so100_v3.py
  2_control_ee_with_pinocchio_so100_real.py  # 兼容入口 -> app.keyboard
  3_vision_grasp.py                            # 兼容入口 -> app.vision
  4_replay_csv_so100.py                        # 兼容入口 -> app.replay
  run_pipeline.py                              # 新统一入口
  FRAMEWORK.md                                 # 框架说明
```

说明：
- 根目录 `2/3/4` 是兼容入口，核心逻辑在 `app/`。
- 离线流程脚本统一在 `pipelines/`。
- 数据与产物分别归档到 `data/` 和 `artifacts/`。

---

## 2. 工作流程（从数据到执行）

### 阶段 A：采集与感知
1. `pipelines/record/1_d435_episode_record.py` 采集 `rgb/depth/timestamps/intrinsics/frame_meta` 到 `data/raw_episodes/`。
2. Dyn-HaMR 生成输出到 `data/hamer_outputs/`。
3. `calibration/outputs/cam2base_latest.json` 作为主标定矩阵，兼容副本在 `record/json/cam2base.json`。

### 阶段 B：控制前处理（ESFP 思路）
1. **Smoothing（已接入基础版）**
   - 代码：`common/common_control.py -> PoseSmoother`
   - 作用：抑制抖动、限制每步最大位移/姿态变化。
2. **Filtering（已接入基础版）**
   - 代码：`TrajectoryGuard + JointCommandFilter`
   - 作用：做工作空间/速度约束与关节命令限速。
3. **Pose-mapping（当前为基础版）**
   - 代码：`CommonRunner.step()` + `SO100Math.build_transform()` + IK
   - 作用：目标末端位姿 -> IK -> 关节命令。

### 阶段 C：执行与记录
1. `CommonRunner.step()` 统一执行控制闭环。
2. 实机接口 `SO100Hardware.send_math_deg_action()` 下发控制。
3. 后续应补齐训练日志导出：`observation + action + timestamp`。

---

## 3. 数学与建模原则（汇报可直接使用）

### 3.1 平滑层（Smoothing）
- 目标：最小化抖动而不破坏轨迹意图。
- 可写成：
  - 数据项：`L_data = ||x_t - x_t^obs||^2`
  - 时间平滑：`L_smooth = ||x_t - x_{t-1}||^2`
  - 骨长一致性：`L_bone = Σ(||b_i(t)|| - l_i)^2`
  - 2D 重投影：`L_2d = ||π(X_t) - u_t||^2`
- 总损失：`L = λ_data L_data + λ_smooth L_smooth + λ_bone L_bone + λ_2d L_2d`

### 3.2 过滤层（Filtering）
- 网络负责高层时序去噪，滤波器负责状态一致性与约束落地。
- 当前实现是工程化限速与限幅；后续可升级到 UKF/EKF：
  - 预测：`x_t^- = f(x_{t-1}, u_{t-1})`
  - 更新：`x_t = x_t^- + K_t(z_t - h(x_t^-))`

### 3.3 映射层（Pose-mapping）
- 人手轨迹到机器人不是直接 IK 即可，需考虑：
  - 工作空间约束
  - 奇异位姿与关节限位
  - 末端方向语义（例如夹爪法向）
- 当前由 `TrajectoryGuard + IK jump guard` 兜底，后续应加入“可达性评分 + 语义映射”。

---

## 4. 面向 VLA 的数据接口（下一阶段核心）

### 为什么从绝对动作改为相对动作
- 绝对坐标强依赖标定和机型尺度，迁移差。
- 相对动作更接近“怎么动”，更符合 UMI 思路。

### 建议标签定义
- 观测：
  - `observation.image_t`
  - `observation.robot_state_t`
  - `observation.hand_state_t`（可选）
- 动作：
  - `action_t = state_{t+1} - state_t`（relative）
  - 或 `action_chunk_t = [action_t ... action_{t+h-1}]`（ACT风格）
- 时间：
  - 使用统一 `timestamp_ns` 做对齐键，禁止依赖“帧序号猜测对齐”。

### 对齐策略（解决“轨迹和视频帧错位”）
1. 以相机时间戳为主时钟。
2. 控制命令记录真实发送时刻。
3. 重采样到固定训练时基（例如 10Hz/20Hz）。
4. 每条样本附 `latency_ms` 或 `delta_t`，训练时可做延迟补偿。

---

## 5. 后续改进路线图（按优先级）

### P0（先做）
1. 增加统一 `TrajectoryLogger`（obs/action/timestamp 全量记录）。
2. 在 CSV 回放链路导出 `relative action + chunk` 训练样本。
3. 把 `T_cam2base`、关节映射、workspace 全部参数化到配置文件。

### P1（核心增强）
1. 平滑层加入骨长一致性 + 2D 重投影约束（离线优化）。
2. 过滤层升级为 UKF（状态+观测噪声可配）。
3. Pose-mapping 增加“可达性/奇异性代价”，输出可执行评分。

### P2（抓取鲁棒性）
1. 引入 GraspNet 作为终端抓取先验，仅在 pinch 低置信时激活。
2. 融合策略：`human_intent_trajectory + grasp_anchor_correction`。
3. 把抓取修正事件记录为训练条件，避免数据污染。

---

## 6. 运行方式

```bash
# 0) 手眼标定采点（相机坐标拾取）
python calibration/1_d435_click.py

# 1) 手眼标定求解（当前脚本内置点对；后续建议改为从 calibration/points 读取）
python calibration/2_handeye_calibration.py
# 说明：求解结果建议同步到
# calibration/outputs/cam2base_latest.json
# record/json/cam2base.json (兼容旧路径)

# 2) D435 数据采集（新路径）
python pipelines/record/1_d435_episode_record.py \
  --root data/raw_episodes/<dataset_name> \
  --fps 30 --width 640 --height 480 \
  --episode-time-s 15 --num-episodes 5

# 2-compat) 采集兼容入口（旧路径命令）
python record/1_d435_episode_record.py --root data/raw_episodes/<dataset_name>

# 3) Dyn-HaMR 推理（在 Dyn-HaMR 工程中执行，输出放到 data/hamer_outputs）
# 这里按你的 Dyn-HaMR 环境命令为准

# 4) 轨迹提取（PKL + episode + 手眼矩阵 -> CSV）
python pipelines/extract/3_extract_traj.py \
  --input-pkl data/hamer_outputs/<run>/<file>.pkl \
  --episode-dir data/raw_episodes/<dataset_name>/episodes/<episode_xxxxxx> \
  --transform-json calibration/outputs/cam2base_latest.json \
  --output-csv data/replay_csv/<name>.csv \
  --fps 30 --hand right --patch-radius 1

# 4-compat) 提取兼容入口
python record/3_extract_traj.py \
  --input-pkl data/hamer_outputs/<run>/<file>.pkl \
  --episode-dir data/raw_episodes/<dataset_name>/episodes/<episode_xxxxxx> \
  --transform-json calibration/outputs/cam2base_latest.json \
  --output-csv data/replay_csv/<name>.csv

# 5) 轨迹可视化验收
python pipelines/visualize/4_visualize_traj.py \
  --input-pkl data/hamer_outputs/<run>/<file>.pkl \
  --episode-dir data/raw_episodes/<dataset_name>/episodes/<episode_xxxxxx> \
  --transform-json calibration/outputs/cam2base_latest.json

# 5-compat) 可视化兼容入口
python record/4_visualize_traj.py \
  --input-pkl data/hamer_outputs/<run>/<file>.pkl \
  --episode-dir data/raw_episodes/<dataset_name>/episodes/<episode_xxxxxx> \
  --transform-json calibration/outputs/cam2base_latest.json

# 新统一入口
python run_pipeline.py --mode keyboard_mirror
python run_pipeline.py --mode keyboard
python run_pipeline.py --mode vision
python run_pipeline.py --mode replay

# 兼容旧命令（仍可用）
python 2_control_ee_with_pinocchio_so100_real.py
python 3_vision_grasp.py
python 4_replay_csv_so100.py
```
