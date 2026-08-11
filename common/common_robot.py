import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import mujoco
import numpy as np
import os
from .src import pinocchio_kinematic as pinocchio_kinematic
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from .config import get_active_robot_profile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCENE_XML_PATH = os.path.join(CURRENT_DIR, 'model/trs_so_arm100/scene.xml')
ARM_XML_PATH = os.path.join(CURRENT_DIR, 'model/trs_so_arm100/so_arm100.xml')


JOINT_NAMES_5DOF = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
]

ACTIVE_ROBOT_PROFILE = get_active_robot_profile()
ACTIVE_ARM_JOINT_NAMES = list(ACTIVE_ROBOT_PROFILE.arm_joint_names)

JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],
    ['shoulder_lift', 2.0, 0.97],
    ['elbow_flex', 0.0, 1.05],
    ['wrist_flex', 0.0, 0.94],
    ['wrist_roll', 0.0, 0.5],
    ['gripper', 0.0, 1.0],
]

DEFAULT_GUESS_Q = np.asarray(ACTIVE_ROBOT_PROFILE.default_guess_q_rad, dtype=float)


@dataclass
class EndEffectorTarget:
    """末端执行器目标 — 使用工具轴方向向量 (tool_axis) 替代 Euler 角。

    工具轴方向 (tool_axis_x/y/z) 是一个单位向量，表示末端执行器
    的朝向（如夹爪的接近方向）。对 5DOF 机械臂，这提供 2 个有效
    姿态约束，与 3 个位置约束合计 5 = 关节数，恰好匹配。
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tool_axis_x: float = 0.0
    tool_axis_y: float = 0.0
    tool_axis_z: float = -1.0   # 默认朝下
    gripper: float = 0.0
    valid: bool = True
    source: str = 'unknown'
    timestamp: Optional[float] = None

    def tool_axis(self) -> np.ndarray:
        """返回归一化的工具轴方向向量 [3]。零向量时返回默认朝下。"""
        a = np.array([self.tool_axis_x, self.tool_axis_y, self.tool_axis_z])
        norm = np.linalg.norm(a)
        if norm < 1e-9:
            return np.array([0.0, 0.0, -1.0])
        return a / norm


class SO100Math:
    @staticmethod
    def raw_to_math(joint_name: str, raw_position: float) -> float:
        for jc in JOINT_CALIBRATION:
            if jc[0] == joint_name:
                return (raw_position - jc[1]) * jc[2]
        return raw_position

    @staticmethod
    def math_to_raw(joint_name: str, math_position: float) -> float:
        for jc in JOINT_CALIBRATION:
            if jc[0] == joint_name:
                return (math_position / jc[2]) + jc[1]
        return math_position



class SO100Kinematics:
    def __init__(self, arm_xml_path: str = ARM_XML_PATH, scene_xml_path: str = SCENE_XML_PATH) -> None:
        self.arm_xml_path = arm_xml_path
        self.scene_xml_path = scene_xml_path
        self.arm = pinocchio_kinematic.Kinematics('JawOffset')
        self.arm.buildFromMJCF(self.arm_xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.last_dof = np.zeros(self.arm.model.nq)
        self.reset_state(DEFAULT_GUESS_Q.copy())

    def reset_state(self, guess_q: np.ndarray) -> None:
        n = min(len(guess_q), self.arm.model.nq)
        self.last_dof[:] = 0.0
        self.last_dof[:n] = guess_q[:n]
        self.data.qpos[:n] = guess_q[:n]
        mujoco.mj_forward(self.model, self.data)

    def fk(self, qpos_rad: np.ndarray) -> np.ndarray:
        return self.arm.fk(qpos_rad)

    def ik(self, p_target: np.ndarray, axis_target: np.ndarray,
           current_q: np.ndarray) -> Tuple[np.ndarray, bool]:
        """5DOF inverse kinematics with tool-axis direction constraint.

        Args:
            p_target:    (3,) target position [x, y, z] in metres.
            axis_target: (3,) target tool-axis direction (unit vector).
            current_q:   (nq,) current joint angles in radians.

        Returns:
            (dof, success) — joint solution and convergence flag.
        """
        dof, info = self.arm.ik(p_target, axis_target,
                                current_arm_motor_q=current_q)
        return dof, info.get("success", False)


class SO100Hardware:
    def __init__(self, port: str = '/dev/ttyACM0', use_degrees: bool = True) -> None:
        cfg = SO100FollowerConfig(port=port, id='single_arm', use_degrees=use_degrees)
        self.robot = SO100Follower(cfg)

    def connect(self) -> None:
        self.robot.connect()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def get_joint_math_deg(self) -> Dict[str, float]:
        obs = self.robot.get_observation()
        out: Dict[str, float] = {}
        for name in ACTIVE_ARM_JOINT_NAMES:
            raw_val = obs.get(f'{name}.pos', 0.0)
            out[name] = SO100Math.raw_to_math(name, raw_val)
        out['gripper'] = obs.get('gripper.pos', 0.0)
        return out

    def get_qpos_rad(self, nq: int) -> np.ndarray:
        joint_math_deg = self.get_joint_math_deg()
        q = np.zeros(nq, dtype=float)
        for i, name in enumerate(ACTIVE_ARM_JOINT_NAMES[:nq]):
            q[i] = math.radians(joint_math_deg[name])
        return q

    def send_math_deg_action(self, joint_math_deg: Dict[str, float], gripper_cmd: float) -> None:
        action = {
            f'{name}.pos': SO100Math.math_to_raw(name, joint_math_deg[name])
            for name in ACTIVE_ARM_JOINT_NAMES
            if name in joint_math_deg
        }
        action['gripper.pos'] = float(gripper_cmd)
        self.robot.send_action(action)

    @staticmethod
    def sim_rad_to_real_action(dof_rad: np.ndarray, gripper_norm: float) -> Dict[str, float]:
        """
        Convert MuJoCo joint radians to real robot action dict.

        Runtime gripper semantic is normalized [0, 1], converted here to
        LeRobot `gripper.pos` in [0, 100] at the hardware boundary.
        """
        sim_deg = [math.degrees(q) for q in dof_rad[:5]]
        real_pan = sim_deg[0] * -1.0 + 3.0
        real_lift = sim_deg[1] * 1.0 + 90.0
        real_elbow = sim_deg[2] * 1.0 - 90.0
        real_wrist = sim_deg[3] * 1.0 + 0.0
        real_roll = sim_deg[4] * 1.0 + 90.0
        return {
            'shoulder_pan.pos': real_pan,
            'shoulder_lift.pos': real_lift,
            'elbow_flex.pos': real_elbow,
            'wrist_flex.pos': real_wrist,
            'wrist_roll.pos': real_roll,
            'gripper.pos': float(np.clip(gripper_norm * 100.0, 0.0, 100.0)),
        }

    def send_sim_rad_action(self, dof_rad: np.ndarray, gripper_norm: float) -> None:
        self.robot.send_action(self.sim_rad_to_real_action(dof_rad, gripper_norm))
