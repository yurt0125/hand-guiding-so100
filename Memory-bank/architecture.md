# Architecture / 架构文档

> 本文件记录 Task-Space Projection（5DOF 投影）改造涉及的模块、接口和数据流。必须与实际代码保持同步。

---

## 项目结构 / Project Structure

```
project_root/
├── common/
│   ├── common_robot.py          # EndEffectorTarget, SO100Kinematics, SO100Hardware
│   ├── common_control.py        # PoseSmoother, TrajectoryGuard, CommonRunner
│   ├── config.py                # 工作空间定义, 机器人配置
│   ├── trajectory_logger.py     # JSONL 日志
│   └── src/
│       └── pinocchio_kinematic.py  # IK/FK 求解器 (Casadi + IPOPT)
├── app/
│   ├── keyboard_mirror_app.py   # 键盘遥操作 + MuJoCo 可视化
│   ├── keyboard_app_refactored.py
│   ├── vision_app_refactored.py # 视觉点击目标
│   └── replay_app_refactored.py # CSV 轨迹回放
├── pipelines/
│   ├── extract/
│   │   └── 3_extract_traj.py    # Dyn-HaMR PKL → 轨迹 CSV
│   ├── hamer/
│   │   └── 4_dynhamr_remote_process.py  # 远程 Dyn-HaMR 处理与结果回传
│   ├── record/
│   │   └── 3_rawdata_record.py  # 单目 D435 原始数据录制（robot_type=hand）
│   ├── export/
│   │   ├── export_vla_dataset.py
│   │   ├── convert_to_lerobot_act.py
│   │   └── build_lerobot_v3_dataset_from_vla.py
│   └── train/
│       └── prepare_act_dataset_from_csv.py
└── Memory-bank/
    ├── implementation_plan.md
    ├── progress.md
    ├── architecture.md (本文件)
    └── agent.md
```

---

## 模块说明 / Module Description

### EndEffectorTarget（数据结构）

**职责:** 统一的末端执行器目标表示，贯穿所有控制模式（键盘/视觉/回放）

**对外接口:**

```python
@dataclass
class EndEffectorTarget:
    # 位置 (3 DOF)
    x: float
    y: float
    z: float
    # 工具轴方向 (2 有效 DOF，存储为 3D 单位向量)
    tool_axis_x: float
    tool_axis_y: float
    tool_axis_z: float  # 默认 -1.0（朝下）
    # 夹爪
    gripper: float      # [0, 1]
    # 元信息
    valid: bool
    source: str
    timestamp: float

    def tool_axis(self) -> np.ndarray:
        """返回归一化的工具轴方向向量 [3]"""
        ...
```

### Kinematics（IK 求解器）

**职责:** 逆运动学求解 — 给定位置 + 工具轴方向，求解关节角度

**对外接口:**

```python
class Kinematics:
    def fk(self, q: np.ndarray) -> np.ndarray:
        """正运动学: 关节角度 → 4x4 齐次变换矩阵"""
        ...

    def ik(self, p_target: np.ndarray, axis_target: np.ndarray,
           current_arm_motor_q: np.ndarray) -> np.ndarray | None:
        """
        5DOF 逆运动学

        参数:
            p_target: [3] 目标位置 (x, y, z)，单位：米
            axis_target: [3] 目标工具轴方向（单位向量）
            current_arm_motor_q: [nq] 当前关节角度，单位：弧度

        返回:
            [nq] 目标关节角度（弧度），或 None（求解失败）

        代价函数:
            min_q  20.0 * ||p(q) - p_target||²
                 + 10.0 * ||a(q) - axis_target||²
                 + 0.005 * ||q - current_q||²
            s.t.  q_min <= q <= q_max
        """
        ...
```

### CommonRunner（控制协调器）

**职责:** 集成平滑 → 滤波 → IK → 关节限速的完整控制管线

**对外接口:**

```python
class CommonRunner:
    def step(self, raw_target: EndEffectorTarget) -> ControlStepResult:
        """
        单步控制

        流程:
            raw_target → PoseSmoother → TrajectoryGuard → IK(p, axis) → JointFilter → 输出

        返回:
            ControlStepResult 包含: raw/smooth/safe target, dof_rad, ik_ok, quality
        """
        ...

    def safe_ik(self, p_target: np.ndarray, axis_target: np.ndarray,
                current_q: np.ndarray) -> tuple[np.ndarray, bool]:
        """带跳变检测的安全 IK 封装"""
        ...
```

### PoseSmoother（平滑器）

**职责:** 对原始目标做 EMA 平滑，降低噪声

**对外接口:**

```python
class PoseSmoother:
    def smooth(self, raw: EndEffectorTarget) -> EndEffectorTarget:
        """
        位置: EMA (alpha_pos=0.25)
        方向: 向量 EMA + 归一化 (alpha_ori=0.20)
        """
        ...
```

### TrajectoryGuard（安全滤波器）

**职责:** 工作空间裁剪 + 速率限制

**对外接口:**

```python
class TrajectoryGuard:
    def apply(self, target: EndEffectorTarget) -> EndEffectorTarget:
        """
        位置: 裁剪到工作空间边界, 限制线速度 (max_xyz_speed=0.18 m/s)
        方向: 限制角速度 (max_axis_angular_speed=100 deg/s, 沿大圆插值)
        """
        ...
```

### 轨迹提取（extract）

**职责:** 从 Dyn-HaMR PKL + 深度图提取机器人轨迹 CSV

**对外接口:**

```python
# 输入: PKL (手部关键点) + 深度 NPZ + cam2base.json
# 输出: CSV 列 = [frame, mid_base_x, mid_base_y, mid_base_z,
#                 tool_axis_x, tool_axis_y, tool_axis_z,
#                 gripper_cmd, hand_found]
```

---

## 接口标准 / Interface Standards

1. 所有公开接口必须有类型注解
2. 所有公开接口必须有文档字符串
3. 模块间调用只能通过本文件定义的接口
4. **姿态表示统一使用工具轴方向向量 `[3]`（单位向量），禁止使用 Euler 角 (roll/pitch/yaw)**
5. **夹爪值统一使用 [0, 1] 归一化表示，仅在发送硬件时转换为 [0, 100]**

---

## 数据流 / Data Flow

```
┌──────────────┐
│ 输入源        │  键盘 / 视觉点击 / CSV 回放 / Dyn-HaMR 提取
│ (x,y,z,axis) │
└──────┬───────┘
       │ EndEffectorTarget
       ▼
┌──────────────┐
│ PoseSmoother │  向量 EMA + 归一化
└──────┬───────┘
       │ EndEffectorTarget (smoothed)
       ▼
┌──────────────┐
│ TrajectoryGuard│ 工作空间裁剪 + 方向角速度限制
└──────┬───────┘
       │ EndEffectorTarget (safe)
       ▼
┌──────────────┐
│  IK 求解器   │  min ||p-p*||² + ||a-a*||²
│ (5DOF投影)   │  5 约束 = 5 关节
└──────┬───────┘
       │ q ∈ R^5 (关节弧度)
       ▼
┌──────────────┐
│JointCmdFilter│ 关节速率限制
└──────┬───────┘
       │ 关节度数命令
       ▼
┌──────────────┐
│ SO100Hardware│  发送到真实机械臂
└──────────────┘
```

---

## 约束与规范 / Constraints

| 约束 | 说明 |
|------|------|
| 5DOF 投影 | IK 只约束位置(3DOF) + 工具轴方向(2DOF)，绕轴自转自由 |
| 无 Euler 角 | 全链路禁止使用 roll/pitch/yaw 表示姿态，消除万向节锁和环绕跳变 |
| 单位向量 | tool_axis 在存储、传输、平滑后必须重新归一化 |
| IK 安全 | IK 失败时返回当前关节位置，不产生跳变 |
| 夹爪语义 | 内部统一 [0,1]，硬件接口处 ×100 |
| CSV 格式 | 新列: tool_axis_x/y/z，旧列 base_roll/pitch/yaw 废弃 |
| 训练数据 | observation = [x,y,z, axis_x,y,z]，action = 相邻帧向量差 |
