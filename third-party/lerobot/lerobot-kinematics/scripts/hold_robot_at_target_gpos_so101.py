#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml-path", required=True, help="Path to scene_so101.xml")
    p.add_argument("--robot-name", default="so101", choices=["so101"])
    # Keep the same base style as the reference script:
    # init_qpos defines the seed / reset pose
    p.add_argument("--init-qpos", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # This is the startup target_gpos you said you want
    p.add_argument("--target-gpos", type=float, nargs=6, default=[0.109, 0.000, 0.120, -1.667, 0.371, 0.000])
    p.add_argument("--max-seconds", type=float, default=1000.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    robot = get_robot(args.robot_name)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml_path)
    qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
    mjdata = mujoco.MjData(mjmodel)

    init_qpos = np.asarray(args.init_qpos, dtype=float)
    if init_qpos.shape != (6,):
        raise ValueError("--init-qpos must have 6 values")

    target_qpos = init_qpos.copy()
    init_gpos = np.asarray(lerobot_FK(init_qpos[1:5], robot=robot), dtype=float)
    target_gpos = np.asarray(args.target_gpos, dtype=float)

    if target_gpos.shape != (6,):
        raise ValueError("--target-gpos must have 6 values")

    # Same spirit as the reference code:
    # start from init_qpos, then use IK to drive toward target_gpos
    mjdata.qpos[qpos_indices] = target_qpos
    mujoco.mj_forward(mjmodel, mjdata)

    target_gpos_last = target_gpos.copy()
    target_qpos_last = target_qpos.copy()

    if args.verbose:
        print(f"[INIT] init_qpos: {[f'{x:.3f}' for x in init_qpos]}")
        print(f"[INIT] init_gpos: {[f'{x:.3f}' for x in init_gpos]}")
        print(f"[INIT] target_gpos: {[f'{x:.3f}' for x in target_gpos]}")
        print("[INFO] No keyboard control and no replay. Robot only moves to target_gpos and holds there.")

    try:
        with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < args.max_seconds:
                step_start = time.time()

                if args.verbose:
                    print("target_gpos:", [f"{x:.3f}" for x in target_gpos])

                fd_qpos = mjdata.qpos[qpos_indices][1:5].copy()
                qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)

                if ik_success:
                    # Same update style as the reference so101 script:
                    # keep joint 0 and gripper from target_qpos, replace middle 4 joints by IK result
                    target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                    mjdata.qpos[qpos_indices] = target_qpos
                    mujoco.mj_step(mjmodel, mjdata)

                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                    viewer.sync()

                    target_gpos_last = target_gpos.copy()
                    target_qpos_last = target_qpos.copy()
                else:
                    target_gpos = target_gpos_last.copy()
                    target_qpos = target_qpos_last.copy()
                    if args.verbose:
                        print("IK fails")

                time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("User interrupted the simulation.")


if __name__ == "__main__":
    main()
