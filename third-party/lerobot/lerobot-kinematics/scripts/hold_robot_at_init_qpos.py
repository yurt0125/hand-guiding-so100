#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from lerobot_kinematics import lerobot_FK, get_robot

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
    p.add_argument("--robot-name", default="so100", choices=["so100", "so101"])
    p.add_argument(
        "--init-qpos",
        type=float,
        nargs=6,
        default=[0.0, -3.14, 3.14, 0.0, -1.57, -0.157],
        help="Initial joint configuration. This is the true startup pose.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def get_joint_names(robot_name: str):
    return JOINT_NAMES_SO100 if robot_name == "so100" else JOINT_NAMES_SO101


def main():
    args = parse_args()

    robot = get_robot(args.robot_name)
    joint_names = get_joint_names(args.robot_name)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml_path)
    mjdata = mujoco.MjData(mjmodel)

    qpos_indices = np.array(
        [mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in joint_names]
    )

    init_qpos = np.asarray(args.init_qpos, dtype=float)
    if init_qpos.shape[0] != 6:
        raise ValueError("--init-qpos must have exactly 6 values")

    # Same initialization idea as the reference code:
    # init_qpos -> FK(init_qpos[1:5]) -> init_gpos / target_gpos
    init_gpos = np.asarray(lerobot_FK(init_qpos[1:5], robot=robot), dtype=float)
    target_qpos = init_qpos.copy()
    target_gpos = init_gpos.copy()

    # Put the robot directly at init_qpos and hold there.
    mjdata.qpos[qpos_indices] = target_qpos
    mujoco.mj_forward(mjmodel, mjdata)

    if args.verbose:
        print(f"[INIT] init_qpos: {[f'{x:.3f}' for x in init_qpos]}")
        print(f"[INIT] target_gpos: {[f'{x:.3f}' for x in target_gpos]}")
        print("[INFO] No replay motion. Robot is held at the startup pose.")

    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
