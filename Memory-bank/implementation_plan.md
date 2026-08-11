# Implementation Plan / 实施计划

> Task-Space Projection：将 6DOF 目标投影到 5DOF 可达空间，解决 SO100 5DOF 机械臂追踪 6DOF 人手姿态的根本性架构问题。

---

## Step 1: 修改数据结构 EndEffectorTarget

**Description / 描述:**
> 将 `EndEffectorTarget` 从 Euler 角表示（roll, pitch, yaw）改为工具轴方向向量表示（tool_axis_x/y/z）。这是所有后续改动的基础数据结构，必须先完成。

**涉及文件:**
- `common/common_robot.py` — `EndEffectorTarget` dataclass

**改动内容:**
```python
# 删除: roll, pitch, yaw 字段
# 新增: tool_axis_x, tool_axis_y, tool_axis_z 字段（单位向量，默认朝下 [0,0,-1]）
# 新增: tool_axis() 方法，返回归一化的 np.ndarray
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`EndEffectorTarget` 不再包含 roll/pitch/yaw 字段
- [ ] 标准2：`tool_axis()` 方法对零向量输入返回默认方向 `[0, 0, -1]`
- [ ] 标准3：`tool_axis()` 返回值的 L2 范数为 1.0（误差 < 1e-6）
- [ ] 标准4：所有直接引用 `.roll` / `.pitch` / `.yaw` 的代码已被定位并标记（后续步骤修改）

---

## Step 2: 修改 IK 求解器 — 旋转约束改为工具轴方向约束

**Description / 描述:**
> 将 `pinocchio_kinematic.py` 中的 IK 代价函数从完整 3DOF 旋转约束改为 2DOF 工具轴方向约束。这是整个投影方案的核心数学改动。同时删除遗留的 `ipdb.set_trace()` 调试断点。

**涉及文件:**
- `common/src/pinocchio_kinematic.py` — `Kinematics.ik()` 方法

**改动内容:**
```python
# 旧接口: ik(T: np.ndarray_4x4, current_arm_motor_q)
# 新接口: ik(p_target: np.ndarray_3, axis_target: np.ndarray_3, current_arm_motor_q)

# 旧代价函数:
#   20.0 * translation_cost + 0.01 * full_rotation_cost + 0.005 * smooth_cost
# 新代价函数:
#   20.0 * translation_cost + 10.0 * axis_direction_cost + 0.005 * smooth_cost

# axis_direction_cost = ||R(q)[:,2] - axis_target||^2
# 其中 R(q)[:,2] 是末端坐标系 z 轴在世界系中的方向（工具轴）
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`ik()` 接口签名变为 `ik(p_target, axis_target, current_arm_motor_q)`
- [ ] 标准2：代价函数中不再包含 `log3` 旋转误差项
- [ ] 标准3：代价函数中包含 `a_actual - axis_target` 的向量差平方项，权重 >= 5.0
- [ ] 标准4：`ipdb.set_trace()` 已删除
- [ ] 标准5：对已知可达目标（如默认 home 位置），IK 求解成功且位置误差 < 1mm、方向误差 < 2°
- [ ] 标准6：对之前因 6DOF 过约束而失败的目标，IK 成功率显著提升

---

## Step 3: 修改 SO100Kinematics 封装层

**Description / 描述:**
> 更新 `SO100Kinematics` 类的 `ik()` 方法，适配新的 IK 求解器接口。不再需要构造完整 4×4 齐次变换矩阵作为 IK 输入。

**涉及文件:**
- `common/common_robot.py` — `SO100Kinematics.ik()` 方法

**改动内容:**
```python
# 旧接口: ik(tf_target: np.ndarray_4x4, current_q) -> np.ndarray
# 新接口: ik(p_target: np.ndarray_3, axis_target: np.ndarray_3, current_q) -> np.ndarray
# 内部直接透传给 pinocchio_kinematic.Kinematics.ik()
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`SO100Kinematics.ik()` 接口接受 `p_target` 和 `axis_target` 两个参数
- [ ] 标准2：不再依赖 `SO100Math.build_transform()` 构造 4×4 矩阵传入 IK
- [ ] 标准3：FK → IK 往返测试：`fk(ik(p, a, q0))` 的位置误差 < 1mm，方向误差 < 2°

---

## Step 4: 修改控制管线 CommonRunner

**Description / 描述:**
> 更新 `CommonRunner.step()` 和 `safe_ik()`，使用新的 5DOF IK 接口。删除 yaw hack（强制使用 actual_yaw）。删除 `build_transform` 调用。

**涉及文件:**
- `common/common_control.py` — `CommonRunner.step()`, `CommonRunner.safe_ik()`

**改动内容:**
```python
# step() 中:
#   删除: actual_yaw = matrix_to_rpy(...); raw_target.yaw = actual_yaw
#   删除: tf_target = build_transform(x, y, z, roll, pitch, yaw)
#   新增: p_target = [safe_target.x, y, z]; axis_target = safe_target.tool_axis()
#   调用: safe_ik(p_target, axis_target, current_q)

# safe_ik() 中:
#   接口从 safe_ik(tf_4x4, current_q) 改为 safe_ik(p_target, axis_target, current_q)
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`step()` 中不再出现 `actual_yaw` 相关代码
- [ ] 标准2：`step()` 中不再调用 `build_transform()`
- [ ] 标准3：`step()` 从 `EndEffectorTarget` 中提取 `tool_axis()` 而非 roll/pitch/yaw
- [ ] 标准4：`safe_ik()` 的跳变检测（max_jump_rad）逻辑保持不变
- [ ] 标准5：IK 失败时仍返回 current_q 并标记 success=False（安全兜底不变）

---

## Step 5: 修改平滑器 PoseSmoother

**Description / 描述:**
> 将姿态平滑从 Euler 角 EMA 改为单位向量 EMA + 归一化。消除 Euler 角环绕跳变问题和万向节锁风险。

**涉及文件:**
- `common/common_control.py` — `PoseSmoother.smooth()`

**改动内容:**
```python
# 删除: self.roll = alpha * raw.roll + (1-alpha) * self.roll (对 pitch 同理)
# 新增: self.axis_smooth = alpha * a_raw + (1-alpha) * self.axis_smooth
# 新增: self.axis_smooth /= norm(self.axis_smooth)  # 重新归一化
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`smooth()` 不再操作 roll/pitch/yaw 字段
- [ ] 标准2：`smooth()` 对 tool_axis 做向量 EMA 后重新归一化
- [ ] 标准3：输出的 `tool_axis()` 始终为单位向量（L2 范数误差 < 1e-6）
- [ ] 标准4：测试：输入从 `[0,0,-1]` 突变到 `[0,0,1]`（180° 翻转），平滑器不产生 NaN 或零向量

---

## Step 6: 修改 TrajectoryGuard 滤波器

**Description / 描述:**
> 更新 `TrajectoryGuard` 和 `FilterCoordinator`，对工具轴方向做速率限制（最大角速度），替代原来对 roll/pitch 的速率限制。

**涉及文件:**
- `common/common_control.py` — `TrajectoryGuard.apply()`, `FilterCoordinator`

**改动内容:**
```python
# 方向速率限制: 计算当前轴与目标轴的夹角
#   angle = arccos(clip(dot(a_current, a_target), -1, 1))
#   如果 angle/dt > max_angular_speed: 沿大圆插值到允许的最大角度
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`TrajectoryGuard` 不再对 roll/pitch/yaw 做独立的速率限制
- [ ] 标准2：方向变化速率被限制在 `max_axis_angular_speed`（建议 100°/s）以内
- [ ] 标准3：Kalman 滤波器（如启用）的状态向量相应更新（roll/pitch → axis_x/axis_y/axis_z）

---

## Step 7: 修改轨迹提取管线

**Description / 描述:**
> 更新 `3_extract_traj.py`，直接输出掌心法向量（tool_axis_x/y/z），不再经过 `normal_to_roll_pitch()` 转换。删除 semantic↔internal 坐标交换逻辑。CSV 列名从 `base_roll/base_pitch/base_yaw` 改为 `tool_axis_x/tool_axis_y/tool_axis_z`。

**涉及文件:**
- `pipelines/extract/3_extract_traj.py`

**改动内容:**
```python
# 删除: normal_to_roll_pitch() 函数
# 删除: semantic_to_internal / internal_to_semantic 坐标交换
# 删除: row["base_roll"], row["base_pitch"], row["base_yaw"]
# 新增: row["tool_axis_x/y/z"] = palm_normal_base（归一化后）
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：输出 CSV 包含列 `tool_axis_x`, `tool_axis_y`, `tool_axis_z`
- [ ] 标准2：输出 CSV 不再包含 `base_roll`, `base_pitch`, `base_yaw` 列
- [ ] 标准3：`normal_to_roll_pitch()` 函数已删除
- [ ] 标准4：`semantic_to_internal_offsets_deg()` 和 `internal_to_semantic_offsets_deg()` 不再被调用
- [ ] 标准5：对已有 PKL 数据重新提取，输出的 tool_axis 均为单位向量（L2 范数误差 < 1e-6）

---

## Step 8: 修改回放应用

**Description / 描述:**
> 更新 `replay_app_refactored.py`，读取新格式 CSV（tool_axis_x/y/z），使用新的 `EndEffectorTarget` 数据结构和 5DOF IK 管线。

**涉及文件:**
- `app/replay_app_refactored.py` — `ReplayController`, `update_target_from_csv()`, `robot_control_loop()`

**改动内容:**
```python
# CSV 读取: 从 base_roll/base_pitch 改为 tool_axis_x/y/z
# ReplayController.apply_pose(): 不再做 roll/pitch 度数→弧度转换，直接设置 tool_axis
# robot_control_loop(): 不再调用 build_transform()，不再做 actual_yaw hack
# 插值: 向量线性插值 + 归一化（替代 Euler 角线性插值）
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：能正确读取新格式 CSV（含 tool_axis_x/y/z 列）
- [ ] 标准2：插值使用向量 LERP + 归一化，不再有 Euler 角环绕跳变
- [ ] 标准3：`--export-action-csv` 导出的关节命令与旧版在可达目标上数值接近（角度差 < 5°）
- [ ] 标准4：端到端回放测试：加载一条新格式 CSV，机械臂运动平滑无跳变

---

## Step 9: 修改键盘控制应用

**Description / 描述:**
> 更新键盘控制应用，将 IJKL 键的姿态控制从"增量 roll/pitch"改为"增量旋转工具轴方向"。

**涉及文件:**
- `app/keyboard_app_refactored.py` — `KeyboardSource`
- `app/keyboard_mirror_app.py` — `KeyboardController`

**改动内容:**
```python
# I/K 键: 绕世界 Y 轴旋转工具轴方向（俯仰）
# J/L 键: 绕世界 X 轴旋转工具轴方向（侧倾）
# 实现: axis_new = Ry(delta) @ axis_current 或 Rx(delta) @ axis_current，然后归一化
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：I/K 键控制工具轴的俯仰方向，操作直觉与旧版一致
- [ ] 标准2：J/L 键控制工具轴的侧倾方向，操作直觉与旧版一致
- [ ] 标准3：连续按键不产生万向节锁或方向突变
- [ ] 标准4：MuJoCo 可视化中末端执行器方向与预期一致

---

## Step 10: 修改视觉点击应用

**Description / 描述:**
> 更新 `vision_app_refactored.py`，点击目标的工具轴方向设为默认朝下 `[0, 0, -1]`（垂直抓取），不再尝试从点击位置推断姿态。

**涉及文件:**
- `app/vision_app_refactored.py` — `VisionTargetSource`

**改动内容:**
```python
# 点击后设置 target.tool_axis_x/y/z = [0, 0, -1]（默认竖直朝下）
# 删除任何对 roll/pitch/yaw 的设置
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：点击后 target 的 tool_axis 为 `[0, 0, -1]`
- [ ] 标准2：机器人移动到点击位置时末端朝下
- [ ] 标准3：不再引用 EndEffectorTarget 的 roll/pitch/yaw

---

## Step 11: 修改训练数据导出管线

**Description / 描述:**
> 更新所有导出脚本，observation/action 空间从 Euler 角切换到工具轴方向向量。确保导出格式与 LeRobot 训练兼容。

**涉及文件:**
- `pipelines/export/export_vla_dataset.py`
- `pipelines/export/convert_to_lerobot_act.py`
- `pipelines/export/build_lerobot_v3_dataset_from_vla.py`
- `pipelines/train/prepare_act_dataset_from_csv.py`
- `common/trajectory_logger.py`

**改动内容:**
```python
# observation state: [x, y, z, tool_axis_x, tool_axis_y, tool_axis_z]  (6D, 维度不变)
# action: [delta_x, delta_y, delta_z, delta_axis_x, delta_axis_y, delta_axis_z]  (6D, 维度不变)
# 注意: delta_axis 需要在应用后重新归一化
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：VLA 表中 state_keys 为 `mid_base_x/y/z, tool_axis_x/y/z`
- [ ] 标准2：action 的相对量计算使用向量差（不再有 Euler 角差）
- [ ] 标准3：导出的 LeRobot 数据集能被 `lerobot-train` 正常加载
- [ ] 标准4：TrajectoryLogger 的 JSONL 输出包含 tool_axis 而非 roll/pitch/yaw
- [ ] 标准5：NaN 出现时记录 warning 日志（不再静默替换为 0）

---

## Step 12: 清理遗留代码和配置

**Description / 描述:**
> 删除所有不再需要的 Euler 角相关代码：`normal_to_roll_pitch()`、`semantic_to_internal_offsets_deg()`、`internal_to_semantic_offsets_deg()`、`build_transform()`（如仅用于 IK 目标构造）、`matrix_to_rpy()`（如仅用于 yaw hack）。更新 `config.py` 中的相关配置。

**涉及文件:**
- `common/config.py` — 删除 semantic/internal 映射函数
- `common/common_robot.py` — 评估 `build_transform` 和 `matrix_to_rpy` 是否还有其他用途
- `pipelines/extract/3_extract_traj.py` — 删除 `normal_to_roll_pitch()`

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`grep -r "roll\|pitch\|yaw" common/ app/` 不返回任何遗留的旧逻辑引用（注释和无关变量除外）
- [ ] 标准2：`semantic_to_internal_offsets_deg()` 和 `internal_to_semantic_offsets_deg()` 已删除
- [ ] 标准3：所有 `import` 语句无未使用的导入
- [ ] 标准4：`config.py` 中无废弃配置项

---

## Step 13: 端到端集成测试

**Description / 描述:**
> 从头跑通完整管线：提取轨迹 → 可视化验证 → 机器人回放 → 导出训练数据。确保所有模块协同工作。

**测试流程:**
1. 用已有的 PKL + 深度数据，运行新版 `3_extract_traj.py`，输出新格式 CSV
2. 用 Rerun 可视化工具验证 tool_axis 方向是否与手掌法向量一致
3. 在 MuJoCo 仿真中回放新格式 CSV，观察运动是否合理
4. （如有硬件）在真实 SO100 上回放，验证运动平滑性
5. 导出训练数据，验证格式正确

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：`3_extract_traj.py` 对已有数据提取成功，无报错
- [ ] 标准2：Rerun 可视化中工具轴方向与手掌朝向一致
- [ ] 标准3：MuJoCo 回放中机械臂运动平滑，无跳变，IK 成功率 > 95%
- [ ] 标准4：导出的 LeRobot 数据集结构正确，`lerobot-train` 可加载
- [ ] 标准5：与旧版对比，相同轨迹的 IK 成功率提升（旧版 < 80% → 新版 > 95%）

---

## Step 14: 末端控制点与姿态耦合问题

**Description / 描述:**
> 处理末端控制点位置对手腕旋转效果的影响。先引入 JawOffset 做末端点测试，再按标定语义需求回退到 JAW 语义点方案。

**涉及文件:**
- `common/model/trs_so_arm100/so_arm100.xml`

**改动内容:**
```python
# 在模型中新增 JawOffset site/frame 用于末端控制测试
# 后续方案回退为：JAW 语义控制点 + 标定显示偏移
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：MuJoCo 中可见 JawOffset 地标并能用于末端控制测试
- [ ] 标准2：切回 JAW 语义点后控制链路正常
- [ ] 标准3：控制点变更不会破坏现有 IK 调用接口

---

## Step 15: 标定点击点与 IK 末端点不一致

**Description / 描述:**
> 标定点击仍以 JAW 为语义点，但显示/计算支持按刚体偏移映射，保证像素点和基座坐标对应一致。

**涉及文件:**
- `calibration/1_dual_camera_robot_click_calibration.py`

**改动内容:**
```python
# 采用刚体变换:
# p_offset_base = R_jaw_base @ p_offset_jaw + t_jaw_base
# 支持 JAW 偏移补偿映射
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：点击 JAW 后可输出偏移映射后的基座坐标
- [ ] 标准2：偏移点可视化与计算结果一致
- [ ] 标准3：标定结果可直接用于后续提取/回放链路

---

## Step 16: keyboard/replay 起点与参数链路修正

**Description / 描述:**
> 修正 replay 起点跳变与参数生效问题，统一新仓库路径下的运行参数链路。

**涉及文件:**
- `run_pipeline.py`
- `app/replay_app_refactored.py`

**改动内容:**
```python
# replay 起始 guess_q 对齐 keyboard 安全位
# --target-offset-x/y/z 改为显式参数生效（默认 0）
# 修复旧仓库路径误用到新仓库路径的问题
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：offset 参数在 replay 中可观测生效
- [ ] 标准2：首帧突跳明显减少
- [ ] 标准3：README 示例路径与实际仓库一致

---

## Step 17: 键盘姿态控制方向统一

**Description / 描述:**
> 统一 I/K/J/L 在 tool_axis 上的旋转方向，提升操作直觉一致性。

**涉及文件:**
- `app/keyboard_app_refactored.py`
- `app/keyboard_mirror_app.py`

**改动内容:**
```python
# I/K/J/L 方向按确认版修正:
# I-，K+，J+，L-
# 并同步更新坐标系/姿态说明注释
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：按键方向与注释一致
- [ ] 标准2：tool_axis 变化与期望一致
- [ ] 标准3：无额外姿态跳变回归

---

## Step 18: 录制与 HaMeR 脚本重命名

**Description / 描述:**
> 统一脚本命名和职责边界，避免流程入口混乱。

**涉及文件:**
- `pipelines/record/3_rawdata_record.py`
- `pipelines/hamer/4_dynhamr_remote_process.py`

**改动内容:**
```python
# 1_lerobot_record_so101_depth.py -> 3_rawdata_record.py
# 8_dynhamr_remote_sync.py -> 4_dynhamr_remote_process.py
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：旧命令迁移到新脚本后可执行
- [ ] 标准2：README 中引用名称与代码一致

---

## Step 19: 数据录制语义统一

**Description / 描述:**
> 统一 raw 录制语义为 hand，并保留深度相机采集链路。

**涉及文件:**
- `pipelines/record/3_rawdata_record.py`

**改动内容:**
```python
# robot_type 统一设为 hand
# 深度相机录制链路保持可用
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：新录制数据 meta/info.json 中 robot_type=hand
- [ ] 标准2：视频与深度数据均可正常生成

---

## Step 20: --resume 与深度录制兼容

**Description / 描述:**
> 修正 resume 后深度分支不稳定问题，保证续录一致性。

**涉及文件:**
- `pipelines/record/3_rawdata_record.py`

**改动内容:**
```python
# 排查并修正 --resume 场景下的深度录制恢复逻辑
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：resume 后深度流持续可用
- [ ] 标准2：续录数据结构与首录一致

---

## Step 21: HaMeR 左右手误判治理

**Description / 描述:**
> 对右手采集链路增加强制右手策略，降低误判导致的数据浪费。

**涉及文件:**
- `pipelines/hamer/4_dynhamr_remote_process.py`
- 远端 Dyn-HaMR/HaMeR 运行入口（run2.py 路线）

**改动内容:**
```python
# 增加 right-hand-only 调用路径
# 使用 run2.py 强制右手处理
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：右手序列误判为左手比例下降
- [ ] 标准2：处理失败时可明确报错并继续下一作业

---

## Step 22: action-图像-episode 严格对齐（核心修复）

**Description / 描述:**
> 重构 sent_action 驱动的数据集生成逻辑，保证动作、图像和分集边界严格一致。

**涉及文件:**
- `pipelines/train/prepare_act_dataset_from_csv.py`

**改动内容:**
```python
# 以 sent_action.csv 为主线
# 按 episode/index 对齐 action、image、episode 边界
# 修复动作错位、分集失效、补帧错位
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：episode 切分与 action 源一致
- [ ] 标准2：replay 的指定 episode 可独立停止
- [ ] 标准3：图像与 action 帧索引对齐可追溯

---

## Step 23: action_valid 合并兼容

**Description / 描述:**
> 处理 action_valid 字段导致的数据集合并不兼容问题。

**涉及文件:**
- `third-party/lerobot/src/lerobot/scripts/lerobot_edit_dataset.py`（调用链）
- 数据集特征定义（输出侧）

**改动内容:**
```python
# 提供 action_valid 兼容方案
# 支持与官方 merge 特征对齐
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：含/不含 action_valid 的数据集可按策略合并
- [ ] 标准2：合并后数据可被训练脚本正常读取

---

## Step 24: Dyn-HaMR 空目录误判成功修复

**Description / 描述:**
> 修复输出目录存在但为空时被误判成功的问题。

**涉及文件:**
- `pipelines/hamer/4_dynhamr_remote_process.py`

**改动内容:**
```python
# 成功判定从“目录存在”改为“关键产物存在且有效”
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：空目录不再判定成功
- [ ] 标准2：缺失 pkl/关键视频时返回失败并可重试

---

## Step 28: HaMeR 多数据集整包合并与命名对齐

**Description / 描述:**
> 新增整包合并脚本，支持将多份 HaMeR 输出按顺序拼接为一份（pkl + render 视频），并保证输出命名与既有提取流程一致。

**涉及文件:**
- `pipelines/hamer/6_merge_hamer_packages.py`
- `README.md`

**改动内容:**
```python
# 新增整包合并脚本：合并 pkl 与 render_all_500.0.mp4
# 支持自定义输出目录 --output-dir 与名称 --merged-name
# 输出命名固定为: {RUN_NAME}_d435_chunk-000_file-000/{RUN_NAME}_d435_chunk-000_file-000.pkl
# pkl 保持原结构，不注入额外字段，不生成额外 manifest
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：两份输入数据集合并后，pkl 帧数等于两者之和，且边界连续
- [ ] 标准2：合并后 render 视频可播放且总帧数与 pkl 对齐
- [ ] 标准3：输出路径与提取命令模板一致，无需改 3_extract_traj.py 既有路径规则

---

## Step 29: Raw 合并参数显式化（保持原命令风格）

**Description / 描述:**
> 在 `6_merge_hamer_packages.py` 中补充 raw 输入/输出显式参数，避免只能按 HaMeR 目录名推断 raw；同时保持原有参数与命名格式兼容不变。

**涉及文件:**
- `pipelines/hamer/6_merge_hamer_packages.py`

**改动内容:**
```python
# 新增 --raw-input-dirs（显式指定 raw 输入数据集）
# 新增 --raw-output-dir（显式指定 raw 合并输出父目录）
# 保留 --raw-root 与旧行为作为兼容 fallback
# 保持输出命名: {merged_name} / {merged_name}_d435_chunk-000_file-000
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：不传 raw 新参数时，旧命令可继续运行
- [ ] 标准2：传入 raw 输入/输出参数时，按指定目录完成 raw 合并
- [ ] 标准3：合并后目录命名与既有提取命令模板保持一致

---

## Step 30: Raw 深度合并低内存化（保持原格式）

**Description / 描述:**
> 参照 `3_merge_depth_episodes.py` 思路，将 `6_merge_hamer_packages.py` 中 raw 深度合并从一次性 `np.concatenate` 改为 `memmap` 流式拼接，降低峰值内存，避免卡死。

**涉及文件:**
- `pipelines/hamer/6_merge_hamer_packages.py`

**改动内容:**
```python
# 去掉 merged_depth_parts 全量缓存
# 改为 episode 逐个写入临时 memmap，再压缩写出 depth.npz
# depth_index.json / episodes/manifest.json 字段与原格式保持一致
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：合并过程峰值内存显著降低，不再因深度拼接卡死
- [ ] 标准2：输出文件结构与字段保持不变（depth.npz/depth_index.json/manifest.json）
- [ ] 标准3：后续 extract/replay 流程可直接复用，无需改命令

---

## Step 31: 避免 /tmp 爆盘（raw 深度临时文件落盘路径修复）

**Description / 描述:**
> 修复 raw 深度合并的临时 `memmap` 默认写入 `/tmp` 导致根分区占满的问题，将临时文件改为写入数据输出目录（`depth/d435`）并在结束后清理。

**涉及文件:**
- `pipelines/hamer/6_merge_hamer_packages.py`

**改动内容:**
```python
# tempfile.mkstemp(..., dir=str(depth_cam_root))
# 临时 mmap 改为写入输出数据目录，避免占用根分区 /tmp
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：合并期间 /tmp 不再出现超大 raw_depth_merge_*.mmap
- [ ] 标准2：根分区不会因该脚本被快速打满
- [ ] 标准3：合并完成后临时 mmap 可被清理，输出格式不变

---

## Step 32: Raw 合并索引语义回退到原格式（frame_index 逐集重置）

**Description / 描述:**
> 修复 raw 合并后 `frame_index` 变为全局连续的问题，恢复为与原数据一致的“每个 episode 从 0 开始”；同时保持 `index` 为全局连续主索引。

**涉及文件:**
- `pipelines/hamer/6_merge_hamer_packages.py`

**改动内容:**
```python
# frame_index = groupby(local_episode_index).cumcount()
# episode_index = episode 全局偏移
# index = 全数据全局连续编号
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：每个 episode 的 frame_index 从 0 开始
- [ ] 标准2：index 保持全局 0..N-1 连续
- [ ] 标准3：下游读取逻辑无需改动

---

## Step 33: Rerun 自动播放默认速率对齐到数据 FPS

**Description / 描述:**
> 修复 `4_visualize_traj_rerun.py` 自动播放过快的问题，默认按数据 FPS（通常 30）播放，而不是默认全速。

**涉及文件:**
- `pipelines/visualize/4_visualize_traj_rerun.py`

**改动内容:**
```python
# --realtime-fps 默认值从 0 改为 -1
# realtime_fps < 0 时自动跟随 --fps
# realtime_fps == 0 时保留“全速播放”语义
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：不传 --realtime-fps 时，自动播放按 --fps（默认 30）进行
- [ ] 标准2：传 --realtime-fps 0 时仍可全速播放
- [ ] 标准3：step-through 模式行为不受影响

---

## Step 34: Rerun 增加当前 Episode 显示

**Description / 描述:**
> 在 `4_visualize_traj_rerun.py` 的状态输出中增加当前 `episode_index`，便于长序列自动播放时快速定位当前处理到哪一集。

**涉及文件:**
- `pipelines/visualize/4_visualize_traj_rerun.py`

**改动内容:**
```python
# 从 CSV 行读取 episode_index 写入 state
# Rerun status/info 面板新增 Episode Index 行
# 终端 Progress 日志追加 episode=...
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：status/info 面板显示当前 Episode Index
- [ ] 标准2：自动播放进度打印包含 episode 编号
- [ ] 标准3：无 CSV episode_index 时可兼容运行

---

## Step 35: Rerun 对无 episode_index 的 CSV 自动推断分集

**Description / 描述:**
> 兼容历史 `*_abs.csv` 不含 `episode_index` 的情况，按 `frame_index` 回绕自动推断 episode 序号，保证状态面板和进度日志能显示“当前集”。

**涉及文件:**
- `pipelines/visualize/4_visualize_traj_rerun.py`

**改动内容:**
```python
# load_extract_csv_rows() 中:
# 若缺 episode_index，则基于 frame_index 递减点进行 episode++ 推断
# 并回写 row['episode_index']
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：旧 CSV（无 episode_index）也能显示 Episode Index
- [ ] 标准2：终端打印提示已启用推断逻辑与推断出的总 episode 数
- [ ] 标准3：对新 CSV（已有 episode_index）行为无影响

---

## Step 36: 提取阶段写入真实 episode_index，并移除可视化推断分集

**Description / 描述:**
> 在 `3_extract_traj.py` 生成 CSV 时直接写入真实 `episode_index`（来源 `depth_index.json`），并删除 `4_visualize_traj_rerun.py` 里按 `frame_index` 回绕推断 episode 的兼容逻辑。

**涉及文件:**
- `pipelines/extract/3_extract_traj.py`
- `pipelines/visualize/4_visualize_traj_rerun.py`

**改动内容:**
```python
# extract: load depth_index.json -> episode_by_depth 映射 -> row['episode_index']
# visualize: 删除 episode 自动推断代码，仅消费 CSV 中的 episode_index
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：新生成 abs.csv 含 episode_index 列
- [ ] 标准2：可视化面板显示 episode 仅依赖 CSV 实际列值
- [ ] 标准3：不存在“全局 frame_index 导致 episode 始终为 0”的假象

---

## Step 37: 轨迹提取增加逐帧进度条

**Description / 描述:**
> 在 `3_extract_traj.py` 的主提取循环中增加进度条，便于观察大序列处理进度与排障。

**涉及文件:**
- `pipelines/extract/3_extract_traj.py`

**改动内容:**
```python
# 新增 tqdm 可选依赖封装 _tqdm()
# build_rows() 主循环显示 extract traj rows 进度
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：运行提取时显示 frame 级进度条
- [ ] 标准2：无 tqdm 环境下可自动回退，不影响功能
- [ ] 标准3：CSV 输出内容与原逻辑一致

---

## Step 38: 轨迹提取进度显示兜底（无 tqdm 仍可见）

**Description / 描述:**
> 解决部分环境下看不到 tqdm 进度条的问题，增加无 tqdm 场景的周期性进度打印。

**涉及文件:**
- `pipelines/extract/3_extract_traj.py`

**改动内容:**
```python
# 若 tqdm 不可用，循环内每 120 帧打印一次 [INFO] extract traj rows: x/N
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：有 tqdm 时显示进度条
- [ ] 标准2：无 tqdm 时显示周期性进度日志
- [ ] 标准3：两种模式输出数据一致

---

## Step 39: Replay reset 阶段支持右方向键提前切集

**Description / 描述:**
> 在 replay 的 episode reset 等待阶段支持手动提前跳过：按右方向键可中断剩余等待时间，立即进入下一集。

**涉及文件:**
- `app/replay_app_refactored.py`

**改动内容:**
```python
# 新增 _skip_reset_event / _is_reset_waiting
# _maybe_wait_for_episode_reset() 等待循环支持 skip 事件
# run_status_loop() 使用 waitKeyEx 监听 RIGHT key 并触发 skip
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：在 RESET_WAIT 状态按右方向键可立即结束等待
- [ ] 标准2：非 RESET_WAIT 状态按右方向键不影响主流程
- [ ] 标准3：reset 阶段不会新增 action 下发和视频帧写入

---

## Step 40: Replay 左键重录“任意时刻有效”+ 视频分段拼接防污染

**Description / 描述:**
> 将 replay 的左方向键重录能力从“仅 reset 阶段”升级为“录制期间任意时刻可触发重录当前集”；同时把视频录制从单文件实时写入改为 episode 分段写入并在结束时拼接，仅保留有效 episode，避免重录前无效帧污染最终视频。

**涉及文件:**
- `app/replay_app_refactored.py`

**改动内容:**
```python
# 左方向键：录制期间任意时刻触发 _redo_prev_episode_event
# 新增 _maybe_handle_immediate_redo_request()：立即回退当前 episode 并清理该集 action/map/video 缓存
# ReplayVideoRecorder 改为 episode 分段写入（*.segments/episode_xxxxxx.mp4）
# stop() 时按保留记录拼接最终 robot.mp4，并重建 map.csv 连续 video_frame_index
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：录制期间任意时刻按左方向键可立即重录当前集
- [ ] 标准2：重录后当前集旧 action 记录与视频映射记录会被删除
- [ ] 标准3：最终 `robot.mp4` 不包含被重录废弃 episode 的无效片段
- [ ] 标准4：`map.csv` 的 `video_frame_index` 在最终输出中连续且与保留 action 一一对应

---

## 变更日志

| 日期 | 变更内容 | 变更人 |
|------|----------|--------|
| 2026-05-13 | 初始计划：Task-Space Projection 5DOF 投影方案，共 13 步 | Guojiabin |
| 2026-05-16 | 完成 Step 14-24 的工程改造与文档同步（控制点、标定、回放、HaMeR、数据对齐、空目录误判修复） | HuangRuida |
| 2026-05-21 | README 新增 YOLO-seg + RealSense(+ICP) 快速测试命令，并同步 Memory-bank 文档 | HuangRuida |
| 2026-05-21 | Rerun 可视化脚本集成 YOLO-seg 可选叠加参数，并同步文档与进度记录 | HuangRuida |
| 2026-05-23 | Replay 模式新增同步视频录制（每 action 一帧）与 map 对齐文件，修复 export-only 判定 | HuangRuida |
| 2026-05-24 | 新增 HaMeR 多数据集整包合并脚本（pkl+视频）并统一输出命名，README 同步更新 | HuangRuida |
| 2026-05-25 | 补充 raw 合并显式参数（--raw-input-dirs/--raw-output-dir），并保持旧命令与命名格式兼容 | HuangRuida |
| 2026-05-25 | Raw 深度合并改为 memmap 流式实现，降低内存峰值并保持原有输出格式 | HuangRuida |
| 2026-05-25 | 修复 raw 深度临时 mmap 路径，避免写入 /tmp 导致根分区爆满 | HuangRuida |
| 2026-05-25 | 修复 raw 合并索引语义：frame_index 按 episode 重置，index 维持全局连续 | HuangRuida |
| 2026-05-25 | 修复 Rerun 自动播放默认速率：未指定 realtime-fps 时跟随数据 fps（30FPS） | HuangRuida |
| 2026-05-25 | Rerun 状态面板与进度日志新增 Episode Index 显示 | HuangRuida |
| 2026-05-25 | Rerun 对无 episode_index 的 CSV 增加自动推断，兼容历史 abs.csv | HuangRuida |
| 2026-05-25 | 提取脚本直接写入真实 episode_index，并移除可视化端 episode 推断逻辑 | HuangRuida |
| 2026-05-25 | 轨迹提取脚本新增逐帧进度条显示（extract traj rows） | HuangRuida |
| 2026-05-25 | 增加提取进度显示兜底：无 tqdm 时周期性打印进度 | HuangRuida |
| 2026-05-25 | Replay reset 阶段新增右方向键提前切集能力（RIGHT key skip wait） | HuangRuida |
| 2026-05-25 | Replay 左键重录升级为录制期间任意时刻生效；视频改为按 episode 分段后拼接，避免重录废片进入最终视频 | HuangRuida |


## Step 41: Replay 新增 resume 续录机制（回退末集重录）

**Description / 描述:**
> 当 replay 中断后，支持按已有导出继续录制：保留已完成 episode，自动回退最后一集并从该集起点重录，避免整段重来。

**涉及文件:**
- `run_pipeline.py`
- `app/replay_app_refactored.py`

**改动内容:**
```python
# 新增 --resume 参数并透传至 replay
# 启动时加载已有 sent_action.csv / map.csv
# 丢弃最后一集记录，row_idx 回退到该集起点继续录
# 视频分段记录与 action 记录同步续写
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：中断后使用 `--resume` 可从最后一集起点续录
- [ ] 标准2：已完成集记录被保留，不会重复录制
- [ ] 标准3：续录后导出的 sent_action.csv 和 map.csv 行序连续
- [ ] 标准4：不加 `--resume` 时行为与原来一致

---


## Step 42: ACT 数据集构建支持机械臂视频对齐写入

**Description / 描述:**
> 在 action-csv 构建 ACT 数据集时，除 HaMeR 人手视频外，增加机械臂回放视频的同步写入能力，用于后续人手→机械臂视觉域对齐训练。

**涉及文件:**
- `pipelines/train/prepare_act_dataset_from_csv.py`

**改动内容:**
```python
# 新增 --robot-video-path
# 在 _build_dataset_from_selected_frames() 中新增 observation.images.robot feature
# 按 sent_action 对齐序列（含 backbone fill 最近邻映射）读取 robot video 帧并写入数据集
```

**验收标准 / Acceptance Criteria:**
- [ ] 标准1：传入 `--robot-video-path` 后输出数据集包含 `observation.images.robot`
- [ ] 标准2：`observation.images.robot` 与 action 序列逐样本对齐
- [ ] 标准3：不传 `--robot-video-path` 时行为与原来一致

---
