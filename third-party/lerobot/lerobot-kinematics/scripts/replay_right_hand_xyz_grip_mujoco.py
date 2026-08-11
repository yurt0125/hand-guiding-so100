#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

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
    p.add_argument("--csv", required=True)
    p.add_argument("--xml-path", required=True)
    p.add_argument("--robot-name", default="so101", choices=["so100", "so101"])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--init-qpos", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p.add_argument("--ee-home", type=float, nargs=3, default=[0.22, 0.00, 0.14])
    p.add_argument("--ee-rpy-fixed", type=float, nargs=3, default=[-1.57, 0.0, 0.0])
    p.add_argument("--pos-scale", type=float, nargs=3, default=[0.8, -0.8, -0.8])
    p.add_argument("--workspace-min", type=float, nargs=3, default=[0.11, -0.08, 0.05])
    p.add_argument("--workspace-max", type=float, nargs=3, default=[0.32, 0.08, 0.24])
    p.add_argument("--gripper-range", type=float, nargs=2, default=[-0.15, 1.0])
    p.add_argument("--alpha-pos", type=float, default=0.25)
    p.add_argument("--alpha-grip", type=float, default=0.2)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def get_joint_names(robot_name: str):
    return JOINT_NAMES_SO101 if robot_name == "so101" else JOINT_NAMES_SO100


def clamp_vec(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def lerp(prev: np.ndarray, cur: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * prev + alpha * cur


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    for c in ["cam_tx", "cam_ty", "cam_tz", "grip_norm_0closed_1open"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.max_frames > 0:
        df = df.iloc[args.start_frame: args.start_frame + args.max_frames].reset_index(drop=True)
    else:
        df = df.iloc[args.start_frame:].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No frames selected from CSV")

    hand_xyz = df[["cam_tx", "cam_ty", "cam_tz"]].to_numpy(dtype=float)
    grip_norm = df["grip_norm_0closed_1open"].to_numpy(dtype=float)

    hand_xyz0 = hand_xyz[0].copy()
    ee_home = np.asarray(args.ee_home, dtype=float)
    ee_rpy_fixed = np.asarray(args.ee_rpy_fixed, dtype=float)
    pos_scale = np.asarray(args.pos_scale, dtype=float)
    ws_min = np.asarray(args.workspace_min, dtype=float)
    ws_max = np.asarray(args.workspace_max, dtype=float)
    gripper_min, gripper_max = map(float, args.gripper_range)

    robot = get_robot(args.robot_name)
    joint_names = get_joint_names(args.robot_name)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml_path)
    qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in joint_names])
    mjdata = mujoco.MjData(mjmodel)

    init_qpos = np.asarray(args.init_qpos, dtype=float)
    target_qpos = init_qpos.copy()
    mjdata.qpos[qpos_indices] = init_qpos
    mujoco.mj_step(mjmodel, mjdata)

    current_chain_q = mjdata.qpos[qpos_indices][1:5].copy()
    last_valid_ee = np.concatenate([ee_home.copy(), ee_rpy_fixed.copy()])
    smooth_ee_xyz = ee_home.copy()

    smooth_g = float(np.clip(grip_norm[0], 0.0, 1.0))
    target_qpos[5] = gripper_min + smooth_g * (gripper_max - gripper_min)

    def frame_to_target(i: int):
        nonlocal smooth_ee_xyz, smooth_g
        delta = hand_xyz[i] - hand_xyz0
        raw_xyz = ee_home + pos_scale * delta
        raw_xyz = clamp_vec(raw_xyz, ws_min, ws_max)
        smooth_ee_xyz = lerp(smooth_ee_xyz, raw_xyz, args.alpha_pos)

        g = float(np.clip(grip_norm[i], 0.0, 1.0))
        smooth_g = (1.0 - args.alpha_grip) * smooth_g + args.alpha_grip * g
        gripper_q = gripper_min + smooth_g * (gripper_max - gripper_min)

        target_gpos = np.concatenate([smooth_ee_xyz.copy(), ee_rpy_fixed.copy()])
        return target_gpos, gripper_q, raw_xyz

    i = 0
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            if i >= len(df):
                if args.loop:
                    i = 0
                    hand_xyz0[:] = hand_xyz[0]
                    smooth_ee_xyz[:] = ee_home
                    smooth_g = float(np.clip(grip_norm[0], 0.0, 1.0))
                else:
                    break

            target_gpos, gripper_q, raw_xyz = frame_to_target(i)
            qpos_inv, ik_success = lerobot_IK(current_chain_q, target_gpos, robot=robot)

            if ik_success:
                target_qpos = np.concatenate(([target_qpos[0]], qpos_inv[:4], [gripper_q]))
                mjdata.qpos[qpos_indices] = target_qpos
                mujoco.mj_step(mjmodel, mjdata)
                current_chain_q = mjdata.qpos[qpos_indices][1:5].copy()
                last_valid_ee = target_gpos.copy()
            else:
                target_gpos = last_valid_ee.copy()

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
            viewer.sync()

            if args.verbose and i % 10 == 0:
                print(
                    f"frame={i:04d} "
                    f"hand=({hand_xyz[i,0]:.3f},{hand_xyz[i,1]:.3f},{hand_xyz[i,2]:.3f}) "
                    f"raw_xyz=({raw_xyz[0]:.3f},{raw_xyz[1]:.3f},{raw_xyz[2]:.3f}) "
                    f"ee_xyz=({target_gpos[0]:.3f},{target_gpos[1]:.3f},{target_gpos[2]:.3f}) "
                    f"grip={gripper_q:.3f} ik={ik_success}"
                )

            i += 1
            dt = time.perf_counter() - step_start
            time.sleep(max(0.0, 1.0 / args.fps - dt))


if __name__ == "__main__":
    main()
