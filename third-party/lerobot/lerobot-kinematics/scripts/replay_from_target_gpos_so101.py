#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from lerobot_kinematics import lerobot_IK, get_robot

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
    p.add_argument("--csv", required=True)
    p.add_argument("--xml-path", required=True)
    p.add_argument("--robot-name", default="so101", choices=["so101"])
    p.add_argument("--frame-col", default="frame_index")
    p.add_argument("--xyz-cols", nargs=3, default=["cam_tx", "cam_ty", "cam_tz"])
    p.add_argument("--grip-col", default="grip_norm_0closed_1open")
    p.add_argument("--pitch-col", default="grasp_pitch_table_deg")

    # Same startup target_gpos style that you verified works
    p.add_argument("--target-gpos", type=float, nargs=6, default=[0.111, 0.000, 0.098, -1.570, 0.400, 0.000])

    # Replay motion relative to the startup target_gpos
    p.add_argument("--pos-scale", type=float, nargs=3, default=[-0.10, -0.10, 0.05])
    p.add_argument("--workspace-min", type=float, nargs=3, default=[0.06, -0.08, 0.07])
    p.add_argument("--workspace-max", type=float, nargs=3, default=[0.20, 0.08, 0.16])

    p.add_argument("--gripper-range", type=float, nargs=2, default=[-0.15, 1.0])

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--alpha-pos", type=float, default=0.20)
    p.add_argument("--alpha-grip", type=float, default=0.15)
    p.add_argument("--alpha-pitch", type=float, default=0.20)

    p.add_argument("--pitch-scale-deg", type=float, default=0.0)
    p.add_argument("--pitch-offset-deg", type=float, default=0.0)
    p.add_argument("--pitch-deg-min", type=float, default=-35.0)
    p.add_argument("--pitch-deg-max", type=float, default=35.0)
    p.add_argument("--freeze-pitch-frames", type=int, default=10)
    p.add_argument("--pitch-outlier-deg", type=float, default=30.0)

    p.add_argument("--startup-hold-seconds", type=float, default=1.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def clamp_vec(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def lerp(prev: np.ndarray, cur: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * prev + alpha * cur


def deg2rad(x: float) -> float:
    return float(np.deg2rad(x))


def maybe_filter_pitch(raw_pitch_deg: float, last_valid_pitch_deg: float, outlier_deg: float):
    if not np.isfinite(raw_pitch_deg):
        return last_valid_pitch_deg, True
    if abs(raw_pitch_deg) > outlier_deg:
        return last_valid_pitch_deg, True
    return raw_pitch_deg, False


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    for c in args.xyz_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required XYZ column: {c}")
    if args.grip_col not in df.columns:
        raise ValueError(f"Missing required grip column: {args.grip_col}")
    if args.pitch_col not in df.columns:
        raise ValueError(f"Missing required pitch column: {args.pitch_col}")
    if args.frame_col not in df.columns:
        df[args.frame_col] = np.arange(len(df), dtype=int)

    hand_xyz = df[args.xyz_cols].to_numpy(dtype=float)
    grip_norm = df[args.grip_col].to_numpy(dtype=float)
    pitch_table_deg = df[args.pitch_col].to_numpy(dtype=float)
    image_frames = df[args.frame_col].to_numpy()

    if len(df) == 0:
        raise ValueError("CSV is empty")

    hand_xyz0 = hand_xyz[0].copy()
    target_gpos_base = np.asarray(args.target_gpos, dtype=float)
    pos_scale = np.asarray(args.pos_scale, dtype=float)
    ws_min = np.asarray(args.workspace_min, dtype=float)
    ws_max = np.asarray(args.workspace_max, dtype=float)
    gripper_min, gripper_max = map(float, args.gripper_range)

    robot = get_robot(args.robot_name)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml_path)
    qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
    mjdata = mujoco.MjData(mjmodel)

    # Start from zeros, exactly like the keycon script, then drive to target_gpos using IK
    target_qpos = np.zeros(6, dtype=float)
    mjdata.qpos[qpos_indices] = target_qpos
    mujoco.mj_forward(mjmodel, mjdata)

    init_grip_norm = float(np.clip(grip_norm[0], 0.0, 1.0)) if np.isfinite(grip_norm[0]) else 0.5
    target_qpos[5] = gripper_min + init_grip_norm * (gripper_max - gripper_min)
    mjdata.qpos[qpos_indices] = target_qpos
    mujoco.mj_forward(mjmodel, mjdata)

    target_gpos = target_gpos_base.copy()
    target_gpos_last = target_gpos.copy()
    target_qpos_last = target_qpos.copy()

    if args.verbose:
        print(f"[INIT] target_gpos: {[f'{x:.3f}' for x in target_gpos]}")
        print("[INFO] Moving to startup target_gpos first, then replaying relative motion.")

    # Move to startup target_gpos first and hold
    startup_deadline = time.time() + args.startup_hold_seconds
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        while viewer.is_running() and time.time() < startup_deadline:
            step_start = time.time()
            fd_qpos = mjdata.qpos[qpos_indices][1:5].copy()
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
            if ik_success:
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos
                mujoco.mj_step(mjmodel, mjdata)
                target_gpos_last = target_gpos.copy()
                target_qpos_last = target_qpos.copy()
            else:
                target_gpos = target_gpos_last.copy()
                target_qpos = target_qpos_last.copy()
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
            viewer.sync()
            dt = mjmodel.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

        smooth_xyz = target_gpos_base[:3].copy()
        base_rpy = target_gpos_base[3:6].copy()
        smooth_g = init_grip_norm
        smooth_pitch_deg = 0.0
        last_valid_pitch_deg = 0.0

        i = 0
        while viewer.is_running():
            step_start = time.time()

            if i >= len(df):
                if args.loop:
                    i = 0
                    hand_xyz0[:] = hand_xyz[0]
                    smooth_xyz[:] = target_gpos_base[:3]
                    smooth_g = init_grip_norm
                    smooth_pitch_deg = 0.0
                    last_valid_pitch_deg = 0.0
                else:
                    break

            delta = hand_xyz[i] - hand_xyz0
            raw_xyz = target_gpos_base[:3] + pos_scale * delta
            raw_xyz = clamp_vec(raw_xyz, ws_min, ws_max)
            smooth_xyz = lerp(smooth_xyz, raw_xyz, args.alpha_pos)

            g = float(np.clip(grip_norm[i], 0.0, 1.0)) if np.isfinite(grip_norm[i]) else smooth_g
            smooth_g = (1.0 - args.alpha_grip) * smooth_g + args.alpha_grip * g
            gripper_q = gripper_min + smooth_g * (gripper_max - gripper_min)

            raw_pitch_deg = float(pitch_table_deg[i]) if np.isfinite(pitch_table_deg[i]) else np.nan
            filtered_pitch_deg, was_filtered = maybe_filter_pitch(
                raw_pitch_deg=raw_pitch_deg,
                last_valid_pitch_deg=last_valid_pitch_deg,
                outlier_deg=args.pitch_outlier_deg,
            )
            if not was_filtered:
                last_valid_pitch_deg = filtered_pitch_deg

            mapped_pitch_deg = filtered_pitch_deg * args.pitch_scale_deg + args.pitch_offset_deg
            mapped_pitch_deg = float(np.clip(mapped_pitch_deg, args.pitch_deg_min, args.pitch_deg_max))
            if i < args.freeze_pitch_frames:
                mapped_pitch_deg = 0.0

            smooth_pitch_deg = (1.0 - args.alpha_pitch) * smooth_pitch_deg + args.alpha_pitch * mapped_pitch_deg

            target_gpos = target_gpos_base.copy()
            target_gpos[:3] = smooth_xyz.copy()
            target_gpos[3:6] = base_rpy.copy()
            target_gpos[4] = base_rpy[1] + deg2rad(smooth_pitch_deg)

            fd_qpos = mjdata.qpos[qpos_indices][1:5].copy()
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)

            if ik_success:
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], [gripper_q]))
                mjdata.qpos[qpos_indices] = target_qpos
                mujoco.mj_step(mjmodel, mjdata)
                target_gpos_last = target_gpos.copy()
                target_qpos_last = target_qpos.copy()
            else:
                target_gpos = target_gpos_last.copy()
                target_qpos = target_qpos_last.copy()

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
            viewer.sync()

            if args.verbose and i % 10 == 0:
                print(
                    f"row={i:04d} "
                    f"img_frame={image_frames[i]} "
                    f"hand=({hand_xyz[i,0]:.3f},{hand_xyz[i,1]:.3f},{hand_xyz[i,2]:.3f}) "
                    f"raw_xyz=({raw_xyz[0]:.3f},{raw_xyz[1]:.3f},{raw_xyz[2]:.3f}) "
                    f"target_gpos=({target_gpos[0]:.3f},{target_gpos[1]:.3f},{target_gpos[2]:.3f},{target_gpos[3]:.3f},{target_gpos[4]:.3f},{target_gpos[5]:.3f}) "
                    f"pitch_table_deg={raw_pitch_deg:.2f} "
                    f"pitch_filtered_deg={filtered_pitch_deg:.2f} "
                    f"pitch_cmd_deg={mapped_pitch_deg:.2f} "
                    f"pitch_smooth_deg={smooth_pitch_deg:.2f} "
                    f"pitch_filtered={was_filtered} "
                    f"grip={gripper_q:.3f} ik={ik_success}"
                )

            i += 1
            dt = max(0.0, 1.0 / args.fps - (time.time() - step_start))
            time.sleep(dt)


if __name__ == "__main__":
    main()
