# Progress / 执行进度

> 本文件记录 Implementation_Plan 中每个步骤的执行状态。每执行完一步必须更新此处。

---

## Step 1: 修改数据结构 EndEffectorTarget

- **Status:** completed
- **Start Time:** 2026-05-13 20:30
- **End Time:** 2026-05-13 20:35
- **备注:** 已将 roll/pitch/yaw 替换为 tool_axis_x/y/z，添加 tool_axis() 方法。已定位全部 ~70 处旧字段引用（分布在 10 个文件中），由后续步骤逐一修改。legacy/ 目录不修改。

---

## Step 2: 修改 IK 求解器 — 旋转约束改为工具轴方向约束

- **Status:** completed
- **Start Time:** 2026-05-13 20:36
- **End Time:** 2026-05-13 20:42
- **备注:** IK 代价函数从 log3 全旋转约束改为工具轴方向向量差约束。权重：位置 20.0 + 方向 10.0 + 平滑 0.005。删除 ipdb.set_trace()。异常处理改为返回 (dof, info) 而非 raise。同时完成 Step 3（SO100Kinematics 封装层同步更新）。

---

## Step 3: 修改 SO100Kinematics 封装层

- **Status:** completed
- **Start Time:** 2026-05-13 20:40
- **End Time:** 2026-05-13 20:42
- **备注:** 与 Step 2 一起完成。ik() 接口改为 (p_target, axis_target, current_q) -> (dof, success)。

---

## Step 4: 修改控制管线 CommonRunner

- **Status:** completed
- **Start Time:** 2026-05-13 20:43
- **End Time:** 2026-05-13 20:50
- **备注:** 与 Step 5、6 一起完成（同文件强耦合）。删除 yaw hack、build_transform 调用。safe_ik() 改用 (p_target, axis_target) 接口。initialize_from_robot() 从 FK z 列提取 tool_axis。

---

## Step 5: 修改平滑器 PoseSmoother

- **Status:** completed
- **Start Time:** 2026-05-13 20:43
- **End Time:** 2026-05-13 20:50
- **备注:** 与 Step 4 一起完成。roll/pitch EMA 改为向量 EMA + slerp 速率限制 + 归一化。

---

## Step 6: 修改 TrajectoryGuard 滤波器

- **Status:** completed
- **Start Time:** 2026-05-13 20:43
- **End Time:** 2026-05-13 20:50
- **备注:** 与 Step 4 一起完成。roll/pitch 速率限制改为 slerp 大圆角速度限制。KalmanPoseFilter 扩展为 12 维状态 [x,y,z,ax,ay,az,v...] + 归一化。

---

## Step 7: 修改轨迹提取管线

- **Status:** completed
- **Start Time:** 2026-05-13 20:51
- **End Time:** 2026-05-13 21:00
- **备注:** CSV 列 base_roll/pitch/yaw → tool_axis_x/y/z。删除 normal_to_roll_pitch()。删除 semantic↔internal 坐标交换。base_normal 直接用 stabilize_normal() 平滑后存储。postprocess_rows_for_training() 更新 core_cols。

---

## Step 8: 修改回放应用

- **Status:** completed
- **Start Time:** 2026-05-14 00:01
- **End Time:** 2026-05-14 00:15
- **备注:** 完全重写。删除 ReplayController.roll/pitch/roll_base_deg/pitch_base_deg → 改用 tool_axis 向量。删除 build_transform()、_matrix_to_rpy()、_infer_csv_angle_unit()、semantic_to_internal 调用。CSV 读取改为 tool_axis_x/y/z。插值改为向量 LERP+归一化。IK 调用改为 ik(p_target, axis_target)。target 元组从 7 元素改为 8 元素。

---

## Step 9: 修改键盘控制应用

- **Status:** completed
- **Start Time:** 2026-05-14 00:16
- **End Time:** 2026-05-14 00:25
- **备注:** 两个文件均完成。keyboard_app_refactored.py 完全重写。keyboard_mirror_app.py 定向修改：删除 matrix_to_rpy、build_transform、yaw hack，IJKL 改为 Rodrigues 小角度旋转工具轴。

---

## Step 10: 修改视觉点击应用

- **Status:** completed
- **Start Time:** 2026-05-14 00:26
- **End Time:** 2026-05-14 00:35
- **备注:** vision_app_refactored.py 无需改动（只操作 x/y/z，tool_axis 从 init_target 继承）。6_validate_click_accuracy.py 修改 EndEffectorTarget 构造（roll/pitch/yaw → tool_axis_x/y/z）。5_replay_csv_mujoco.py 完全重写：ReplayRow 改为 tool_axis，删除 angle_unit 推断，删除 REPLAY_ROLL/PITCH_BASE_DEG，IK 使用新接口。

---

## Step 11: 修改训练数据导出管线

- **Status:** completed
- **Start Time:** 2026-05-14 00:36
- **End Time:** 2026-05-14 00:50
- **备注:** 5 个文件全部更新。trajectory_logger: target.roll/pitch/yaw → tool_axis_x/y/z。export_vla_dataset + convert_to_lerobot_act + build_lerobot_v3: DEFAULT_STATE_COLUMNS/KEYS 更新。convert_to_lerobot_act 新增 NaN warning。prepare_act_dataset_from_csv: REQUIRED_CSV_COLS 更新，IK 模拟改用新接口，删除 _build_transform/yaw hack/semantic-internal 映射，插值改为向量 LERP。

---

## Step 12: 清理遗留代码和配置

- **Status:** completed
- **Start Time:** 2026-05-14 00:51
- **End Time:** 2026-05-14 00:58
- **备注:** 删除：config.py 中 REPLAY_ROLL/PITCH_BASE_DEG、SEMANTIC_*_SIGN_RULE、semantic_to_internal_offsets_deg、internal_to_semantic_offsets_deg。common_robot.py 中 SO100Math.matrix_to_rpy、SO100Math.build_transform。3_extract_traj.py 中 wrap_angle_deg、stabilize_angle。prepare_act_dataset_from_csv.py 中 _build_transform、_matrix_to_rpy、_infer_csv_angle_unit。legacy/ 未修改。4_visualize_traj.py 保留（独立可视化脚本，可后续单独更新）。

---

## Step 13: 端到端集成测试

- **Status:** pending
- **Start Time:**
- **End Time:**
- **备注:** 最终验收，依赖所有前置步骤

---

## 汇总 / Summary

| 步骤 | 状态 | 完成时间 |
|------|------|----------|
| Step 1: 修改数据结构 | completed | 2026-05-13 |
| Step 2: 修改 IK 求解器 | completed | 2026-05-13 |
| Step 3: 修改 Kinematics 封装层 | completed | 2026-05-13 |
| Step 4: 修改 CommonRunner | completed | 2026-05-13 |
| Step 5: 修改平滑器 | completed | 2026-05-13 |
| Step 6: 修改 TrajectoryGuard | completed | 2026-05-13 |
| Step 7: 修改轨迹提取 | completed | 2026-05-14 |
| Step 8: 修改回放应用 | completed | 2026-05-14 |
| Step 9: 修改键盘控制 | completed | 2026-05-14 |
| Step 10: 修改视觉点击 | completed | 2026-05-14 |
| Step 11: 修改训练数据导出 | completed | 2026-05-14 |
| Step 12: 清理遗留代码 | completed | 2026-05-14 |
| Step 13: 端到端集成测试 | pending | - |

**Overall Progress:** 92% (12/13 completed)

---


## 2026-05-16 文档与工程更新

---

## Step 14: 末端控制点与姿态耦合问题

- **Status:** completed
- **Start Time:** 2026-05-15 10:00
- **End Time:** 2026-05-15 10:40
- **备注:** 在 `so_arm100.xml` 新增 `JawOffset` 做末端控制测试；后续按需求回退到“JAW 语义控制点 + 标定显示偏移”方案。

---

## Step 15: 标定点击点与 IK 末端点不一致

- **Status:** completed
- **Start Time:** 2026-05-15 10:45
- **End Time:** 2026-05-15 11:30
- **备注:** 修改 `calibration/1_dual_camera_robot_click_calibration.py`，点击仍对 JAW，显示/计算用刚体偏移映射；采用 `p_offset_base = R_jaw_base * p_offset_jaw + t_jaw_base`。

---

## Step 16: keyboard/replay 起点与参数链路修正

- **Status:** completed
- **Start Time:** 2026-05-15 11:35
- **End Time:** 2026-05-15 12:30
- **备注:** replay 起始 `guess_q`（起始点坐标：`x=-0.013, y=-0.171, z=0.120` 对齐 keyboard 安全位，安全位关节坐标：`q=[-0.1124, -3.0792, 2.98, 0.6003, -1.4801, 0.0]`）；`--target-offset-x/y/z` 改为显式生效（默认 0）；修复旧仓库路径混用到新仓库的问题。

---

## Step 17: 键盘姿态控制方向统一

- **Status:** completed
- **Start Time:** 2026-05-15 13:30
- **End Time:** 2026-05-15 14:00
- **备注:** 键盘姿态控制改为：`I/K` 改 `tool_axis` 围绕世界 `X` 轴的旋转分量，`J/L` 改 `tool_axis` 围绕世界 `Y` 轴的旋转分量；按键方向确认版为 `I-，K+，J+，L-`；末端姿态和位姿的坐标系示意图及坐标正负说明已同步补充。
---

## Step 18: 录制与 HaMeR 脚本重命名

- **Status:** completed
- **Start Time:** 2026-05-15 14:05
- **End Time:** 2026-05-15 14:25
- **备注:** `1_lerobot_record_so101_depth.py -> 3_rawdata_record.py`；`8_dynhamr_remote_sync.py -> 4_dynhamr_remote_process.py`。

---

## Step 19: 数据录制语义统一

- **Status:** completed
- **Start Time:** 2026-05-15 14:30
- **End Time:** 2026-05-15 14:45
- **备注:** `3_rawdata_record.py` 中 robot_type 统一设为 hand，并保留深度相机录制链路。

---

## Step 20: --resume 与深度录制兼容

- **Status:** completed
- **Start Time:** 2026-05-15 15:00
- **End Time:** 2026-05-15 15:40
- **备注:** 排查并修正 resume 后深度分支恢复逻辑，保证继续录制时深度数据不丢。

---

## Step 21: HaMeR 左右手误判治理

- **Status:** completed
- **Start Time:** 2026-05-15 16:00
- **End Time:** 2026-05-15 17:10
- **备注:** 远端流程增加右手强制方案（run2.py 路线），`4_dynhamr_remote_process.py` 支持 right-hand-only 调用，减少误判废片。

---

## Step 22: action-图像-episode 严格对齐（核心修复）

- **Status:** completed
- **Start Time:** 2026-05-15 17:20
- **End Time:** 2026-05-15 20:30
- **备注:** 重构 `prepare_act_dataset_from_csv.py` 对齐逻辑，以 sent_action.csv 为主线，按 episode/index 严格对齐动作、图像和分集边界，修复错位与分集失效问题。

---

## Step 23: action_valid 合并兼容

- **Status:** completed
- **Start Time:** 2026-05-15 20:35
- **End Time:** 2026-05-15 21:00
- **备注:** 给出 action_valid 兼容方案用于官方 merge；后续按需求保留向官方原格式对齐的路径。

---

## Step 24: Dyn-HaMR 空目录误判成功修复

- **Status:** completed
- **Start Time:** 2026-05-16 00:45
- **End Time:** 2026-05-16 01:05
- **备注:** 修复“输出目录存在但为空仍判成功”的问题，改为检查关键产物存在后再判成功。

---

## Step 25: README 增补 YOLO-seg 测试命令与文档同步

- **Status:** completed
- **Start Time:** 2026-05-21 00:20
- **End Time:** 2026-05-21 00:30
- **备注:** 在 `README.md` 新增 `pipelines/vision/test_yolo_seg_realsense.py` 的标准运行命令（`--model /home/rita/hand-guiding-so100-master/models/yolov8n-seg.pt`，目标类 `banana`），并按 `agent.md` 要求同步更新 `implementation_plan.md` 与 `progress.md`。

---

## Step 26: Rerun 可视化集成 YOLO-seg 叠加

- **Status:** completed
- **Start Time:** 2026-05-21 00:35
- **End Time:** 2026-05-21 00:55
- **备注:** 在 `pipelines/visualize/4_visualize_traj_rerun.py` 增加 YOLO-seg 可选开关与参数（`--enable-yolo-seg`、`--yolo-model`、`--yolo-target-class`、`--yolo-conf`、`--yolo-device`），并在状态面板增加 `YOLO Target Count`。README 同步新增运行示例。

---

## Step 27: Replay 同步录制视频与 action 帧级对应

- **Status:** completed
- **Start Time:** 2026-05-23 10:30
- **End Time:** 2026-05-23 11:20
- **备注:** 在 `run_pipeline.py` 与 `app/replay_app_refactored.py` 新增 replay 录制参数透传与同步录制逻辑。录制改为“每下发一条 action 抓取一帧并写入 MP4”，并输出 `*.map.csv`（`video_frame_index/video_timestamp_s/action_seq/csv_row_index/abs_index`）用于严格对齐；同时修复 `--export-action-csv` 与真机端口同时使用时误进入 export-only 的问题。

---

## Step 28: HaMeR 多数据集整包合并与命名对齐

- **Status:** completed
- **Start Time:** 2026-05-24 22:20
- **End Time:** 2026-05-24 23:55
- **备注:** 新增 `pipelines/hamer/6_merge_hamer_packages.py`，支持多数据集整包合并（pkl+render 视频，带进度条），并将输出命名对齐为 `${RUN_NAME}_d435_chunk-000_file-000`，保证后续 `3_extract_traj.py` 可直接复用原命令模板。`pkl` 保持原结构，不额外注入字段。

---

## Step 29: Raw 合并参数显式化（保持原命令风格）

- **Status:** completed
- **Start Time:** 2026-05-25 00:10
- **End Time:** 2026-05-25 00:30
- **备注:** 在 `pipelines/hamer/6_merge_hamer_packages.py` 新增 `--raw-input-dirs` 与 `--raw-output-dir`，可显式指定 raw 输入/输出；同时保留 `--raw-root` 与原推断逻辑作为兼容 fallback。输出目录与命名保持原格式：`${merged_name}` 与 `${merged_name}_d435_chunk-000_file-000`。

---

## Step 30: Raw 深度合并低内存化（保持原格式）

- **Status:** completed
- **Start Time:** 2026-05-25 00:35
- **End Time:** 2026-05-25 01:00
- **备注:** 参照 `3_merge_depth_episodes.py`，将 `6_merge_hamer_packages.py` 中 raw 深度合并从 `np.concatenate` 全量拼接改为 `memmap` 流式写入，显著降低峰值内存；`depth.npz`、`depth_index.json`、`episodes/manifest.json` 输出格式保持不变。

---

## Step 31: 避免 /tmp 爆盘（raw 深度临时文件落盘路径修复）

- **Status:** completed
- **Start Time:** 2026-05-25 14:20
- **End Time:** 2026-05-25 14:35
- **备注:** 将 `6_merge_hamer_packages.py` 的临时 `memmap` 文件从默认 `/tmp` 改为写入 `depth/d435` 输出目录，避免根分区被临时文件占满；清理后根分区从 100% 降至 77%。

---

## Step 32: Raw 合并索引语义回退到原格式（frame_index 逐集重置）

- **Status:** completed
- **Start Time:** 2026-05-25 14:40
- **End Time:** 2026-05-25 14:55
- **备注:** 修复 `6_merge_hamer_packages.py` 中 raw 合并索引逻辑：`frame_index` 改回每个 episode 内从 0 开始；`episode_index` 继续全局偏移；`index` 继续全局连续，保持与原始数据语义一致。

---

## Step 33: Rerun 自动播放默认速率对齐到数据 FPS

- **Status:** completed
- **Start Time:** 2026-05-25 15:00
- **End Time:** 2026-05-25 15:10
- **备注:** 修改 `pipelines/visualize/4_visualize_traj_rerun.py`：`--realtime-fps` 默认从 `0` 改为 `-1`，自动播放时 `realtime_fps < 0` 将跟随 `--fps`（默认 30FPS）；`--realtime-fps 0` 仍代表全速播放。

---

## Step 34: Rerun 增加当前 Episode 显示

- **Status:** completed
- **Start Time:** 2026-05-25 15:12
- **End Time:** 2026-05-25 15:20
- **备注:** 更新 `pipelines/visualize/4_visualize_traj_rerun.py`：从 CSV 读取 `episode_index` 写入状态；在 Rerun `status/info` 面板新增 `Episode Index` 行；自动播放 `Progress` 日志追加 `episode=...`。

---

## Step 35: Rerun 对无 episode_index 的 CSV 自动推断分集

- **Status:** completed
- **Start Time:** 2026-05-25 15:25
- **End Time:** 2026-05-25 15:35
- **备注:** 在 `load_extract_csv_rows()` 增加兼容逻辑：若 `episode_index` 缺失，则根据 `frame_index` 回绕自动推断 episode；启动时打印推断摘要，保证历史 abs.csv 也能显示当前 episode。

---

## Step 36: 提取阶段写入真实 episode_index，并移除可视化推断分集

- **Status:** completed
- **Start Time:** 2026-05-25 15:40
- **End Time:** 2026-05-25 15:55
- **备注:** 修改 `3_extract_traj.py`：从 `depth_index.json` 构建 depth-frame 到 episode 的映射并写入 CSV `episode_index` 列；修改 `4_visualize_traj_rerun.py`：删除按 `frame_index` 回绕推断 episode 的临时兼容逻辑。

---

## Step 37: 轨迹提取增加逐帧进度条

- **Status:** completed
- **Start Time:** 2026-05-25 16:00
- **End Time:** 2026-05-25 16:10
- **备注:** 在 `3_extract_traj.py` 中新增 `tqdm` 可选封装并应用到 `build_rows()` 主循环，运行时显示 `extract traj rows` 帧级进度；无 `tqdm` 环境自动回退为普通循环。

---

## Step 38: 轨迹提取进度显示兜底（无 tqdm 仍可见）

- **Status:** completed
- **Start Time:** 2026-05-25 16:15
- **End Time:** 2026-05-25 16:25
- **备注:** 在 `3_extract_traj.py` 主循环增加无 tqdm 场景的进度打印兜底（每 120 帧一次），避免某些终端/环境下进度不可见。

---

## Step 39: Replay reset 阶段支持右方向键提前切集

- **Status:** completed
- **Start Time:** 2026-05-25 16:30
- **End Time:** 2026-05-25 16:45
- **备注:** 修改 `app/replay_app_refactored.py`：`RESET_WAIT` 状态新增右方向键提前跳过等待逻辑（`waitKeyEx` 监听 RIGHT key），按键后立即继续下一集；仅在 reset 等待时生效。

---

## Step 40: Replay 左键重录改为“任意时刻有效”并修复视频污染

- **Status:** completed
- **Start Time:** 2026-05-25 16:50
- **End Time:** 2026-05-25 17:25
- **备注:** 修改 `app/replay_app_refactored.py`：左方向键从“仅 reset 阶段”扩展为“录制期间任意时刻可重录当前集”；并将回放视频录制改为按 episode 临时分段，最终仅拼接保留 episode，避免重录前无效片段污染最终 `robot.mp4`。


---

## Step 41: Replay 新增 resume 续录机制（回退末集重录）

- **Status:** completed
- **Start Time:** 2026-05-25 17:35
- **End Time:** 2026-05-25 18:05
- **备注:** 为 `run_pipeline.py --mode replay` 增加 `--resume`；启动时读取已存在 `sent_action.csv`（及可选 map.csv），自动保留已完成集并回退到“最后一集起点”重录，支持中断后续录；视频分段记录同步续写。


---

## Step 42: ACT 数据集构建支持机械臂视频对齐写入

- **Status:** completed
- **Start Time:** 2026-05-25 18:10
- **End Time:** 2026-05-25 18:30
- **备注:** 修改 `pipelines/train/prepare_act_dataset_from_csv.py`，新增 `--robot-video-path`；在 action-csv 模式下按 sent_action 对齐序列将机械臂视频写入 `observation.images.robot`，与 `observation.images.d435` 同步进入输出训练集。
