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
    p.add_argument("--ee-home", type=float, nargs=3, default=[0.111, 0.000, 0.098])
    p.add_argument("--ee-rpy-fixed", type=float, nargs=3, default=[-1.57, 0.0, 0.0])

    # NEW: axis remap instead of assuming robot xyz <- cam xyz
    # Example: --xyz-map z x y means:
    #   robot_x <- cam_z
    #   robot_y <- cam_x
    #   robot_z <- cam_y
    p.add_argument("--xyz-map", nargs=3, choices=["x", "y", "z"], default=["x", "y", "z"])
    # NEW: sign and scale are applied after remap
    p.add_argument("--xyz-sign", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p.add_argument("--xyz-scale", type=float, nargs=3, default=[0.20, 0.20, 0.10])

    p.add_argument("--workspace-min", type=float, nargs=3, default=[0.10, -0.05, 0.09])
    p.add_argument("--workspace-max", type=float, nargs=3, default=[0.18, 0.05, 0.14])

    p.add_argument("--gripper-range", type=float, nargs=2, default=[-0.15, 1.0])
    p.add_argument("--alpha-pos", type=float, default=0.20)
    p.add_argument("--alpha-grip", type=float, default=0.15)
    p.add_argument("--alpha-pitch", type=float, default=0.20)

    p.add_argument("--xyz-cols", nargs=3, default=["cam_tx", "cam_ty", "cam_tz"])
    p.add_argument("--grip-col", default="grip_norm_0closed_1open")
    p.add_argument("--pitch-col", default="grasp_pitch_table_deg")
    p.add_argument("--frame-col", default="frame_index")

    p.add_argument("--pitch-scale-deg", type=float, default=1.0)
    p.add_argument("--pitch-offset-deg", type=float, default=0.0)
    p.add_argument("--pitch-deg-min", type=float, default=-35.0)
    p.add_argument("--pitch-deg-max", type=float, default=35.0)
    p.add_argument("--freeze-pitch-frames", type=int, default=0)
    p.add_argument("--pitch-outlier-deg", type=float, default=30.0)

    # NEW: start-up settling frames
    p.add_argument("--startup-settle-steps", type=int, default=30)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def get_joint_names(robot_name: str):
    return JOINT_NAMES_SO101 if robot_name == "so101" else JOINT_NAMES_SO100


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


def remap_delta(delta_cam: np.ndarray, xyz_map: list[str], xyz_sign: np.ndarray, xyz_scale: np.ndarray) -> np.ndarray:
    idx = {"x": 0, "y": 1, "z": 2}
    remapped = np.array([delta_cam[idx[a]] for a in xyz_map], dtype=float)
    return xyz_sign * xyz_scale * remapped


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

    if args.max_frames > 0:
        df = df.iloc[args.start_frame: args.start_frame + args.max_frames].reset_index(drop=True)
    else:
        df = df.iloc[args.start_frame:].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No frames selected from CSV")

    hand_xyz = df[args.xyz_cols].to_numpy(dtype=float)
    grip_norm = df[args.grip_col].to_numpy(dtype=float)
    pitch_table_deg = df[args.pitch_col].to_numpy(dtype=float)
    image_frames = df[args.frame_col].to_numpy()

    hand_xyz0 = hand_xyz[0].copy()
    ee_home = np.asarray(args.ee_home, dtype=float)
    ee_rpy_fixed = np.asarray(args.ee_rpy_fixed, dtype=float)
    xyz_sign = np.asarray(args.xyz_sign, dtype=float)
    xyz_scale = np.asarray(args.xyz_scale, dtype=float)
    ws_min = np.asarray(args.workspace_min, dtype=float)
    ws_max = np.asarray(args.workspace_max, dtype=float)
    gripper_min, gripper_max = map(float, args.gripper_range)

    robot = get_robot(args.robot_name)
    joint_names = get_joint_names(args.robot_name)

    mjmodel = mujoco.MjModel.from_xml_path(args.xml_path)
    qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in joint_names])
    mjdata = mujoco.MjData(mjmodel)

    init_qpos = np.asarray(args.init_qpos, dtype=float)

    # Seed with provided init-qpos first
    mjdata.qpos[qpos_indices] = init_qpos
    mujoco.mj_forward(mjmodel, mjdata)

    qpos_before = mjdata.qpos[qpos_indices].copy()

    # ===== More complete startup init =====
    # Solve IK at ee-home pose, then write the whole controllable chain at start.
    init_target_gpos = np.concatenate([ee_home.copy(), ee_rpy_fixed.copy()])
    seed_chain_q = mjdata.qpos[qpos_indices][1:5].copy()
    qpos_init_ik, init_ik_success = lerobot_IK(seed_chain_q, init_target_gpos, robot=robot)
    if not init_ik_success:
        raise RuntimeError(
            "Failed to initialize robot at ee-home / ee-rpy-fixed. "
            "Try adjusting --ee-home, --ee-rpy-fixed, or --init-qpos."
        )

    startup_qpos = mjdata.qpos[qpos_indices].copy()
    # keep joint 0 as user init unless you know how to solve it too
    startup_qpos[1:5] = qpos_init_ik[:4]
    startup_qpos[5] = gripper_min + float(np.clip(grip_norm[0], 0.0, 1.0)) * (gripper_max - gripper_min)

    mjdata.qpos[qpos_indices] = startup_qpos
    for _ in range(max(1, args.startup_settle_steps)):
        mujoco.mj_step(mjmodel, mjdata)

    qpos_after = mjdata.qpos[qpos_indices].copy()

    current_chain_q = mjdata.qpos[qpos_indices][1:5].copy()
    last_valid_ee = init_target_gpos.copy()

    smooth_ee_xyz = ee_home.copy()
    smooth_g = float(np.clip(grip_norm[0], 0.0, 1.0))
    smooth_pitch_deg = 0.0
    last_valid_pitch_deg = 0.0

    if args.verbose:
        print(
            "[INIT] start target_gpos="
            f"({init_target_gpos[0]:.3f}, {init_target_gpos[1]:.3f}, {init_target_gpos[2]:.3f}, "
            f"{init_target_gpos[3]:.3f}, {init_target_gpos[4]:.3f}, {init_target_gpos[5]:.3f}) "
            f"init_ik={init_ik_success}"
        )
        print(f"[INIT] qpos_before={np.round(qpos_before, 4).tolist()}")
        print(f"[INIT] qpos_after ={np.round(qpos_after, 4).tolist()}")
        print(
            "[INIT] xyz remap: "
            f"robot_xyz <- ({args.xyz_map[0]}, {args.xyz_map[1]}, {args.xyz_map[2]}) "
            f"with sign={args.xyz_sign} scale={args.xyz_scale}"
        )

    def frame_to_target(i: int):
        nonlocal smooth_ee_xyz, smooth_g, smooth_pitch_deg, last_valid_pitch_deg

        delta_cam = hand_xyz[i] - hand_xyz0
        delta_robot = remap_delta(delta_cam, args.xyz_map, xyz_sign, xyz_scale)
        raw_xyz = ee_home + delta_robot
        raw_xyz = clamp_vec(raw_xyz, ws_min, ws_max)
        smooth_ee_xyz = lerp(smooth_ee_xyz, raw_xyz, args.alpha_pos)

        g = float(np.clip(grip_norm[i], 0.0, 1.0))
        smooth_g = (1.0 - args.alpha_grip) * smooth_g + args.alpha_grip * g
        gripper_q = gripper_min + smooth_g * (gripper_max - gripper_min)

        raw_pitch_deg = float(pitch_table_deg[i])
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

        ee_rpy = ee_rpy_fixed.copy()
        ee_rpy[1] = ee_rpy_fixed[1] + deg2rad(smooth_pitch_deg)

        target_gpos = np.concatenate([smooth_ee_xyz.copy(), ee_rpy])
        return target_gpos, gripper_q, raw_xyz, delta_cam, delta_robot, raw_pitch_deg, filtered_pitch_deg, mapped_pitch_deg, smooth_pitch_deg, was_filtered

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
                    smooth_pitch_deg = 0.0
                    last_valid_pitch_deg = 0.0
                else:
                    break

            target_gpos, gripper_q, raw_xyz, delta_cam, delta_robot, raw_pitch_deg, filtered_pitch_deg, mapped_pitch_deg, smooth_pitch_deg_now, was_filtered = frame_to_target(i)
            qpos_inv, ik_success = lerobot_IK(current_chain_q, target_gpos, robot=robot)

            if ik_success:
                next_qpos = mjdata.qpos[qpos_indices].copy()
                next_qpos[1:5] = qpos_inv[:4]
                next_qpos[5] = gripper_q
                mjdata.qpos[qpos_indices] = next_qpos
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
                    f"row={i:04d} "
                    f"img_frame={image_frames[i]} "
                    f"hand=({hand_xyz[i,0]:.3f},{hand_xyz[i,1]:.3f},{hand_xyz[i,2]:.3f}) "
                    f"delta_cam=({delta_cam[0]:.3f},{delta_cam[1]:.3f},{delta_cam[2]:.3f}) "
                    f"delta_robot=({delta_robot[0]:.3f},{delta_robot[1]:.3f},{delta_robot[2]:.3f}) "
                    f"raw_xyz=({raw_xyz[0]:.3f},{raw_xyz[1]:.3f},{raw_xyz[2]:.3f}) "
                    f"ee_xyz=({target_gpos[0]:.3f},{target_gpos[1]:.3f},{target_gpos[2]:.3f}) "
                    f"pitch_table_deg={raw_pitch_deg:.2f} "
                    f"pitch_filtered_deg={filtered_pitch_deg:.2f} "
                    f"pitch_cmd_deg={mapped_pitch_deg:.2f} "
                    f"pitch_smooth_deg={smooth_pitch_deg_now:.2f} "
                    f"pitch_filtered={was_filtered} "
                    f"grip={gripper_q:.3f} ik={ik_success}"
                )

            i += 1
            dt = time.perf_counter() - step_start
            time.sleep(max(0.0, 1.0 / args.fps - dt))


if __name__ == "__main__":
    main()
