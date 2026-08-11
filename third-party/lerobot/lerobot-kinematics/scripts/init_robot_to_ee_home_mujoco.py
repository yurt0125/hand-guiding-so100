#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from lerobot_kinematics import lerobot_IK, get_robot

JOINT_NAMES_SO101 = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

JOINT_NAMES_SO100 = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml-path", required=True)
    p.add_argument("--robot-name", default="so101", choices=["so100", "so101"])

    p.add_argument("--init-qpos", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 你想要的末端初始位姿
    p.add_argument("--ee-home", type=float, nargs=3, default=[0.111, 0.000, 0.098])
    p.add_argument("--ee-rpy-fixed", type=float, nargs=3, default=[-1.570, 0.0, 0.0])

    # 夹爪初始开合
    p.add_argument("--gripper-q", type=float, default=0.697)

    # 初始化后静置多少个仿真 step
    p.add_argument("--settle-steps", type=int, default=50)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def get_joint_names(robot_name: str):
    return JOINT_NAMES_SO101 if robot_name == "so101" else JOINT_NAMES_SO100


def main():
    args = parse_args()

    robot = get_robot(args.robot_name)
    joint_names = get_joint_names(args.robot_name)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml_path)
    mjdata = mujoco.MjData(mjmodel)

    qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in joint_names])

    init_qpos = np.asarray(args.init_qpos, dtype=float)
    mjdata.qpos[qpos_indices] = init_qpos
    mujoco.mj_forward(mjmodel, mjdata)

    qpos_before = mjdata.qpos[qpos_indices].copy()

    target_gpos = np.concatenate([
        np.asarray(args.ee_home, dtype=float),
        np.asarray(args.ee_rpy_fixed, dtype=float),
    ])

    seed_chain_q = mjdata.qpos[qpos_indices][1:5].copy()
    qpos_inv, ik_success = lerobot_IK(seed_chain_q, target_gpos, robot=robot)

    if not ik_success:
        raise RuntimeError(
            "Failed to initialize robot at ee-home / ee-rpy-fixed.\n"
            "Try adjusting --ee-home, --ee-rpy-fixed, or --init-qpos."
        )

    startup_qpos = mjdata.qpos[qpos_indices].copy()
    # 保留第 0 个关节不动，只设置 IK 解出的 4 个中间关节和夹爪
    startup_qpos[1:5] = qpos_inv[:4]
    startup_qpos[5] = float(args.gripper_q)

    mjdata.qpos[qpos_indices] = startup_qpos

    for _ in range(max(1, args.settle_steps)):
        mujoco.mj_step(mjmodel, mjdata)

    qpos_after = mjdata.qpos[qpos_indices].copy()

    if args.verbose:
        print(
            f"[INIT] target_gpos=({target_gpos[0]:.3f}, {target_gpos[1]:.3f}, {target_gpos[2]:.3f}, "
            f"{target_gpos[3]:.3f}, {target_gpos[4]:.3f}, {target_gpos[5]:.3f})"
        )
        print(f"[INIT] init_ik={ik_success}")
        print(f"[INIT] qpos_before={np.round(qpos_before, 4).tolist()}")
        print(f"[INIT] qpos_after ={np.round(qpos_after, 4).tolist()}")

    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
