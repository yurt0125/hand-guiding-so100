#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
import signal
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config import VISION_WORKSPACE
from common.src.pinocchio_kinematic import Kinematics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARM_XML_PATH = str(PROJECT_ROOT / "common" / "model" / "trs_so_arm100" / "so_arm100.xml")
INTERRUPTED = False


REQUIRED_CSV_COLS = [
    "mid_base_x",
    "mid_base_y",
    "mid_base_z",
    "tool_axis_x",
    "tool_axis_y",
    "tool_axis_z",
    "gripper_cmd",
]

REPLAY_Y_MAX_THRESHOLD = -0.15
REPLAY_MAX_POSITION_STEP_M = 0.008
REPLAY_MAX_ANGLE_STEP_RAD = math.radians(2.0)
REPLAY_MAX_CANDIDATE_JOINT_JUMP_RAD = 0.9



def _prepare_csv(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for c in REQUIRED_CSV_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    raw_valid = np.isfinite(df[REQUIRED_CSV_COLS]).all(axis=1).to_numpy()
    if "hand_found" in df.columns:
        hand_found = pd.to_numeric(df["hand_found"], errors="coerce").fillna(0).to_numpy() > 0
        raw_valid = raw_valid & hand_found

    # Always make a complete sequence so every RGB frame has an action/state.
    df[REQUIRED_CSV_COLS] = df[REQUIRED_CSV_COLS].interpolate(limit_direction="both").ffill().bfill()
    if df[REQUIRED_CSV_COLS].isna().any().any():
        raise ValueError("CSV still contains NaN after interpolation/fill.")

    return df, raw_valid.astype(np.float32)


def _make_ee_vectors(
    csv_df: pd.DataFrame,
    target_offset_x: float,
    target_offset_y: float,
    target_offset_z: float,
) -> np.ndarray:
    out = np.zeros((len(csv_df), 7), dtype=np.float32)
    out[:, 0] = csv_df["mid_base_x"].to_numpy(dtype=np.float32) + np.float32(target_offset_x)
    out[:, 1] = csv_df["mid_base_y"].to_numpy(dtype=np.float32) + np.float32(target_offset_y)
    out[:, 2] = csv_df["mid_base_z"].to_numpy(dtype=np.float32) + np.float32(target_offset_z)
    out[:, 3] = csv_df["tool_axis_x"].to_numpy(dtype=np.float32)
    out[:, 4] = csv_df["tool_axis_y"].to_numpy(dtype=np.float32)
    out[:, 5] = csv_df["tool_axis_z"].to_numpy(dtype=np.float32)
    out[:, 6] = csv_df["gripper_cmd"].to_numpy(dtype=np.float32)
    return out


def _make_qpos_vectors(
    csv_df: pd.DataFrame,
    target_offset_x: float,
    target_offset_y: float,
    target_offset_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build qpos-like 6D actions:
      [q0, q1, q2, q3, q4, gripper]
    Returns:
      action_vectors, ik_valid_mask
    """
    arm = Kinematics("JawOffset")
    arm.buildFromMJCF(ARM_XML_PATH)

    q_last = np.array([0.0547, -0.4193, 0.6309, -0.2114, -1.5357, 0.0], dtype=np.float64)
    out = np.zeros((len(csv_df), 6), dtype=np.float32)
    valid = np.zeros((len(csv_df),), dtype=np.float32)

    x_min, x_max = float(VISION_WORKSPACE["x_min"]), float(VISION_WORKSPACE["x_max"])
    y_min, y_max = float(VISION_WORKSPACE["y_min"]), float(VISION_WORKSPACE["y_max"])
    z_min, z_max = float(VISION_WORKSPACE["z_min"]), float(VISION_WORKSPACE["z_max"])

    def _normalize_axis(a):
        norm = np.linalg.norm(a)
        return a / norm if norm > 1e-9 else np.array([0.0, 0.0, -1.0])

    def _slerp_axis(a, b, max_angle):
        angle = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
        if angle < 1e-9:
            return a.copy()
        if angle <= max_angle:
            return b.copy()
        t = max_angle / angle
        return _normalize_axis((1.0 - t) * a + t * b)

    def _lerp_axis(a, b, alpha):
        return _normalize_axis((1.0 - alpha) * a + alpha * b)

    # Replay-like online state (target latch + rate-limited controller pose).
    first_row = csv_df.iloc[0]
    ctrl_x = float(np.clip(float(first_row["mid_base_x"]) + target_offset_x, x_min, x_max))
    ctrl_y = float(np.clip(float(first_row["mid_base_y"]) + target_offset_y, y_min, y_max))
    ctrl_z = float(np.clip(float(first_row["mid_base_z"]) + target_offset_z, z_min, z_max))
    ctrl_axis = _normalize_axis(np.array([
        float(first_row["tool_axis_x"]),
        float(first_row["tool_axis_y"]),
        float(first_row["tool_axis_z"]),
    ]))

    target_x, target_y, target_z = ctrl_x, ctrl_y, ctrl_z
    target_axis = ctrl_axis.copy()
    target_gripper = float(np.clip(first_row["gripper_cmd"], 0.0, 1.0))

    # previous_valid_target: (x, y, z, ax, ay, az, gripper)
    previous_valid_target: tuple | None = None
    pending_invalid = 0

    for i, row in tqdm(
        csv_df.iterrows(),
        total=len(csv_df),
        desc="IK(qpos) replay-like",
        unit="frame",
        dynamic_ncols=True,
    ):
        if INTERRUPTED:
            raise KeyboardInterrupt("Interrupted by user")

        row_x = float(np.clip(float(row["mid_base_x"]) + float(target_offset_x), x_min, x_max))
        row_y = float(np.clip(float(row["mid_base_y"]) + float(target_offset_y), y_min, y_max))
        row_z = float(np.clip(float(row["mid_base_z"]) + float(target_offset_z), z_min, z_max))
        row_axis = _normalize_axis(np.array([
            float(row["tool_axis_x"]), float(row["tool_axis_y"]), float(row["tool_axis_z"]),
        ]))
        row_gripper = float(np.clip(row["gripper_cmd"], 0.0, 1.0))

        # Replay y-threshold skip behavior.
        if row_y > (REPLAY_Y_MAX_THRESHOLD + 1e-6):
            pending_invalid += 1
        else:
            # Reachability gate with joint-jump check (5DOF IK).
            p_cand = np.array([row_x, row_y, row_z])
            reachable = False
            try:
                cand_dof, cand_info = arm.ik(p_cand, row_axis, current_arm_motor_q=q_last)
                if cand_info.get("success", False):
                    cand = np.asarray(cand_dof, dtype=np.float64).reshape(-1)
                    ref = np.asarray(q_last, dtype=np.float64).reshape(-1)
                    compare_count = min(5, cand.shape[0], ref.shape[0])
                    jump_max = float(np.max(np.abs(cand[:compare_count] - ref[:compare_count])))
                    reachable = jump_max <= REPLAY_MAX_CANDIDATE_JOINT_JUMP_RAD
            except Exception:
                reachable = False

            if not reachable:
                pending_invalid += 1
            else:
                # Interpolate skipped spans like replay.
                if previous_valid_target is not None and pending_invalid > 0:
                    px, py, pz, pax, pay, paz, pg = previous_valid_target
                    prev_axis = _normalize_axis(np.array([pax, pay, paz]))
                    alpha = 1.0 / float(pending_invalid + 1)
                    target_x = (1.0 - alpha) * px + alpha * row_x
                    target_y = (1.0 - alpha) * py + alpha * row_y
                    target_z = (1.0 - alpha) * pz + alpha * row_z
                    target_axis = _lerp_axis(prev_axis, row_axis, alpha)
                    target_gripper = (1.0 - alpha) * pg + alpha * row_gripper
                else:
                    target_x, target_y, target_z = row_x, row_y, row_z
                    target_axis = row_axis.copy()
                    target_gripper = row_gripper
                previous_valid_target = (row_x, row_y, row_z, float(row_axis[0]), float(row_axis[1]), float(row_axis[2]), row_gripper)
                pending_invalid = 0

        # Replay-like controller rate limit per frame.
        ctrl_x = _limit_step(ctrl_x, target_x, REPLAY_MAX_POSITION_STEP_M)
        ctrl_y = _limit_step(ctrl_y, target_y, REPLAY_MAX_POSITION_STEP_M)
        ctrl_z = _limit_step(ctrl_z, target_z, REPLAY_MAX_POSITION_STEP_M)
        ctrl_axis = _slerp_axis(ctrl_axis, target_axis, REPLAY_MAX_ANGLE_STEP_RAD)

        # 5DOF IK: position + tool axis direction
        p_target = np.array([ctrl_x, ctrl_y, ctrl_z])
        try:
            ik_ret = arm.ik(p_target, ctrl_axis, current_arm_motor_q=q_last)
            dof, info = ik_ret
            ok = info.get("success", False) if isinstance(info, dict) else bool(info)
            if dof is None or not ok:
                raise RuntimeError("ik returned invalid result")
            dof = np.asarray(dof, dtype=np.float64).reshape(-1)
            q_last = dof[:6].copy()
            out[i, :5] = q_last[:5].astype(np.float32)
            out[i, 5] = np.float32(target_gripper)
            valid[i] = 1.0
        except Exception:
            # Hold-last fallback to keep frame alignment complete.
            out[i, :5] = q_last[:5].astype(np.float32)
            out[i, 5] = np.float32(target_gripper)
            valid[i] = 0.0

    return out, valid



def _limit_step(current_value: float, target_value: float, max_step: float) -> float:
    delta = target_value - current_value
    if delta > max_step:
        return current_value + max_step
    if delta < -max_step:
        return current_value - max_step
    return target_value


def _convert_qpos_to_replay_real_degrees(vectors: np.ndarray) -> np.ndarray:
    """
    Convert IK qpos vectors to replay-consistent real-space degrees.

    Pipeline:
      sim(qpos rad) -> replay-mapped real(deg)

    Input expects:
      vectors[:, :5] in radians, vectors[:, 5] in [0, 1].
    """
    out = np.asarray(vectors, dtype=np.float32).copy()

    sim_deg = np.degrees(out[:, :5])
    # Must stay consistent with app/replay_app_refactored.py and common/common_robot.py.
    real_pan = sim_deg[:, 0] * -1.0 + 3.0
    real_lift = sim_deg[:, 1] * +1.0 + 90.0
    real_elbow = sim_deg[:, 2] * +1.0 - 90.0
    real_wrist = sim_deg[:, 3] * +1.0 + 0.0
    real_roll = sim_deg[:, 4] * +1.0 + 90.0
    real_deg = np.stack([real_pan, real_lift, real_elbow, real_wrist, real_roll], axis=1)

    out[:, :5] = real_deg
    out[:, 5] = np.clip(out[:, 5] * 100.0, 0.0, 100.0)
    return out


def _video_num_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _replace_rgb_video_with_hamer_render(
    output_root: Path,
    hamer_render_video: Path,
    expected_frames: int,
    allow_mismatch: bool,
) -> None:
    if not hamer_render_video.exists():
        raise FileNotFoundError(f"hamer-render-video not found: {hamer_render_video}")

    src_frames = _video_num_frames(hamer_render_video)
    if (not allow_mismatch) and src_frames != expected_frames:
        raise ValueError(
            f"HaMeR video frame count mismatch: video={src_frames}, expected={expected_frames}. "
            "Set --allow-hamer-video-frame-mismatch to bypass."
        )

    target_video_path = output_root / "videos" / "observation.images.d435" / "chunk-000" / "file-000.mp4"
    if not target_video_path.exists():
        raise FileNotFoundError(
            f"Target LeRobot RGB video not found at {target_video_path}. "
            "This helper currently expects single-camera single-file layout."
        )
    shutil.copy2(hamer_render_video, target_video_path)
    print(f"[INFO] replaced RGB video with HaMeR render: {target_video_path}")
    print(f"[INFO] HaMeR video frames: {src_frames}, expected dataset frames: {expected_frames}")


def _patch_stats_for_added_features(dataset_root: Path) -> None:
    """
    Ensure stats.json contains action/state/action_valid for training-time normalization.
    """
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.utils import load_stats, write_stats

    data_dir = dataset_root / "data"
    parquet_paths = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    actions = []
    states = []
    valids = []
    for parquet_path in parquet_paths:
        frame_df = pd.read_parquet(parquet_path, columns=["action", "observation.state", "action_valid"])
        actions.append(np.vstack(frame_df["action"].to_list()).astype(np.float32))
        states.append(np.vstack(frame_df["observation.state"].to_list()).astype(np.float32))
        valids.append(np.vstack(frame_df["action_valid"].to_list()).astype(np.float32))

    action_arr = np.concatenate(actions, axis=0)
    state_arr = np.concatenate(states, axis=0)
    valid_arr = np.concatenate(valids, axis=0)

    stats = load_stats(dataset_root) or {}
    stats["action"] = get_feature_stats(action_arr, axis=0, keepdims=False)
    stats["observation.state"] = get_feature_stats(state_arr, axis=0, keepdims=False)
    stats["action_valid"] = get_feature_stats(valid_arr, axis=0, keepdims=False)
    write_stats(stats, dataset_root)


def _resolve_action_csv_indices(
    action_df: pd.DataFrame,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    required_action_cols = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    missing_cols = [c for c in required_action_cols if c not in action_df.columns]
    if missing_cols:
        raise ValueError(f"Action CSV missing required columns: {missing_cols}")

    if "abs_index" in action_df.columns:
        idx_col = "abs_index"
    elif "csv_row_index" in action_df.columns:
        idx_col = "csv_row_index"
    elif "frame_index" in action_df.columns:
        idx_col = "frame_index"
    else:
        idx_col = None

    if idx_col is None:
        if len(action_df) != num_frames:
            raise ValueError(
                f"Action CSV rows ({len(action_df)}) must equal dataset num_frames ({num_frames}) "
                "when no frame index column exists."
            )
        source_indices = np.arange(num_frames, dtype=np.int64)
        vectors = action_df[required_action_cols].to_numpy(dtype=np.float32)
        return source_indices, vectors

    indexed = action_df[[idx_col] + required_action_cols].copy()
    indexed[idx_col] = pd.to_numeric(indexed[idx_col], errors="coerce")
    indexed = indexed.dropna(subset=[idx_col])
    indexed[idx_col] = indexed[idx_col].astype(int)
    indexed = indexed[(indexed[idx_col] >= 0) & (indexed[idx_col] < num_frames)]
    if indexed.empty:
        raise ValueError(f"Action CSV has no valid {idx_col} rows in [0, {num_frames - 1}].")

    # Keep original sent_action row order and duplicates.
    indexed = indexed.reset_index(drop=True)
    source_indices = indexed[idx_col].to_numpy(dtype=np.int64)
    vectors = indexed[required_action_cols].to_numpy(dtype=np.float32)
    return source_indices, vectors


def _nonempty_pkl_field(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set, str, bytes)):
        return len(value) > 0
    try:
        arr = np.asarray(value)
        return arr.size > 0
    except Exception:
        return True


def _load_pkl_valid_indices(pkl_path: Path, expected_total_frames: int) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        records = [payload[k] for k in payload.keys()]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported PKL payload type: {type(payload)}")

    if len(records) != expected_total_frames:
        raise ValueError(
            f"PKL frame count ({len(records)}) != expected total frames ({expected_total_frames})."
        )

    valid = np.zeros((len(records),), dtype=bool)
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        valid[i] = (
            _nonempty_pkl_field(rec.get("mano"))
            or _nonempty_pkl_field(rec.get("cam_trans"))
            or _nonempty_pkl_field(rec.get("tracked_ids"))
            or _nonempty_pkl_field(rec.get("tid"))
        )
    return np.flatnonzero(valid).astype(np.int64)


def _build_image_abs_alignment(source_indices: np.ndarray, valid_abs_indices: np.ndarray) -> np.ndarray:
    if len(valid_abs_indices) == 0:
        raise ValueError("PKL valid index set is empty.")
    sorted_valid = np.asarray(valid_abs_indices, dtype=np.int64)
    pos = np.searchsorted(sorted_valid, source_indices, side="left")
    out = np.empty_like(source_indices, dtype=np.int64)
    for i, src_idx in enumerate(source_indices.tolist()):
        if pos[i] < len(sorted_valid) and sorted_valid[pos[i]] == src_idx:
            out[i] = src_idx
            continue
        prev_idx = sorted_valid[pos[i] - 1] if pos[i] > 0 else None
        next_idx = sorted_valid[pos[i]] if pos[i] < len(sorted_valid) else None
        if prev_idx is not None:
            out[i] = int(prev_idx)
        elif next_idx is not None:
            out[i] = int(next_idx)
        else:
            out[i] = int(src_idx)
    return out


def _build_dataset_from_selected_frames(
    src_ds,
    source_indices: np.ndarray,
    action_vectors: np.ndarray,
    output_root: Path,
    repo_id: str,
    valid_abs_indices: np.ndarray,
    image_video_path: Path | None = None,
    robot_video_path: Path | None = None,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if len(source_indices) == 0:
        raise ValueError("No selected frame indices for dataset build.")
    if len(source_indices) != len(action_vectors):
        raise ValueError(
            f"Selected frame count ({len(source_indices)}) != action vector count ({len(action_vectors)})."
        )

    base_features = {k: v.copy() if isinstance(v, dict) else v for k, v in src_ds.meta.features.items()}
    vec_info = {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    }
    valid_info = {"dtype": "float32", "shape": (1,), "names": ["is_valid"]}
    base_features["action"] = vec_info
    base_features["observation.state"] = vec_info
    base_features["action_valid"] = valid_info
    if robot_video_path is not None:
        if "observation.images.d435" not in base_features:
            raise KeyError("source dataset missing feature 'observation.images.d435', cannot infer robot image feature.")
        robot_img_info = dict(base_features["observation.images.d435"])
        base_features["observation.images.robot"] = robot_img_info

    new_ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=src_ds.fps,
        features=base_features,
        root=output_root,
        robot_type=src_ds.meta.robot_type,
        use_videos=len(src_ds.meta.video_keys) > 0,
        batch_encoding_size=1,
    )

    data_parts = sorted((Path(src_ds.root) / "data").glob("chunk-*/file-*.parquet"))
    if not data_parts:
        raise FileNotFoundError(f"No data parquet found under {(Path(src_ds.root) / 'data')}")
    raw_map = pd.concat(
        [pd.read_parquet(p, columns=["index", "episode_index", "frame_index", "task_index"]) for p in data_parts],
        ignore_index=True,
    ).drop_duplicates(subset=["index"], keep="first")
    row_df = pd.DataFrame(
        {
            "row_order": np.arange(len(source_indices), dtype=np.int64),
            "index": source_indices.astype(np.int64),
        }
    )
    action_df = pd.DataFrame(action_vectors, columns=vec_info["names"])
    row_df = pd.concat([row_df, action_df], axis=1)
    sent_df = row_df.merge(raw_map, on="index", how="left")
    if sent_df[["episode_index", "frame_index", "task_index"]].isna().any().any():
        bad = sent_df[sent_df["episode_index"].isna()].head(10)
        raise RuntimeError(f"Some action rows cannot map to raw index. sample:\n{bad.to_string(index=False)}")
    sent_df["episode_index"] = sent_df["episode_index"].astype(int)
    sent_df["frame_index"] = sent_df["frame_index"].astype(int)
    sent_df["task_index"] = sent_df["task_index"].astype(int)
    sent_df = sent_df.sort_values("row_order").reset_index(drop=True)

    # Episode backbone: all HaMeR-valid abs indices inside each raw episode.
    valid_set = set(valid_abs_indices.tolist())
    backbone_df = raw_map[raw_map["index"].isin(valid_set)].copy()
    backbone_df["episode_index"] = backbone_df["episode_index"].astype(int)
    backbone_df["frame_index"] = backbone_df["frame_index"].astype(int)
    backbone_df["task_index"] = backbone_df["task_index"].astype(int)

    timeline_rows: list[dict] = []
    for episode_idx, ep_sent in sent_df.groupby("episode_index", sort=True):
        ep_sent = ep_sent.copy().reset_index(drop=True)
        ep_back = backbone_df[backbone_df["episode_index"] == int(episode_idx)].copy()
        sent_unique_idx = set(ep_sent["index"].astype(int).tolist())
        missing_back = ep_back[~ep_back["index"].isin(sent_unique_idx)].copy()

        ep_sent["is_backbone_fill"] = False
        ep_sent["source"] = "sent_action"
        ep_sent["sort_key"] = ep_sent["index"].astype(np.int64) * 1000 + np.arange(len(ep_sent), dtype=np.int64)
        ep_sent["robot_action_seq"] = ep_sent["row_order"].astype(np.int64)

        if len(missing_back) > 0 and len(ep_sent) > 0:
            sent_indices = ep_sent["index"].to_numpy(dtype=np.int64)
            sent_actions = ep_sent[vec_info["names"]].to_numpy(dtype=np.float32)
            filled_rows = []
            for _, back_row in missing_back.iterrows():
                idx = int(back_row["index"])
                d = np.abs(sent_indices - idx)
                nearest_pos = int(np.argmin(d))
                action_vec = sent_actions[nearest_pos]
                row = {
                    "row_order": int(ep_sent["row_order"].max()) + 1,
                    "index": idx,
                    "episode_index": int(back_row["episode_index"]),
                    "frame_index": int(back_row["frame_index"]),
                    "task_index": int(back_row["task_index"]),
                    "is_backbone_fill": True,
                    "source": "hamer_backbone_fill",
                    "sort_key": int(idx) * 1000 + 999,
                    "robot_action_seq": int(ep_sent.iloc[nearest_pos]["row_order"]),
                }
                for j, name in enumerate(vec_info["names"]):
                    row[name] = float(action_vec[j])
                filled_rows.append(row)
            ep_fill = pd.DataFrame(filled_rows)
            ep_mix = pd.concat([ep_sent, ep_fill], ignore_index=True)
        else:
            ep_mix = ep_sent
        ep_mix = ep_mix.sort_values("sort_key").reset_index(drop=True)
        timeline_rows.append(ep_mix)

    if len(timeline_rows) == 0:
        raise RuntimeError("No rows available after episode alignment.")
    src_df = pd.concat(timeline_rows, ignore_index=True)

    idx_series = src_df["index"].to_numpy(dtype=np.int64)
    valid_set = set(valid_abs_indices.tolist())
    src_df["source_is_hamer_valid"] = src_df["index"].isin(valid_set)
    src_df["image_abs_index"] = _build_image_abs_alignment(idx_series, valid_abs_indices)
    source_counts = pd.Series(idx_series).value_counts()
    src_df["source_dup_count"] = src_df["index"].map(source_counts).astype(int)
    src_df["is_replay_extra"] = (~src_df["source_is_hamer_valid"]) | (src_df["source_dup_count"] > 1)
    valid_rank = {int(v): i for i, v in enumerate(valid_abs_indices.tolist())}
    src_df["image_frame_index"] = src_df["image_abs_index"].map(valid_rank)
    if src_df["image_frame_index"].isna().any():
        raise RuntimeError("Failed to map image_abs_index to HaMeR video frame index.")
    src_df["image_frame_index"] = src_df["image_frame_index"].astype(int)
    if "robot_action_seq" in src_df.columns:
        src_df["robot_action_seq"] = src_df["robot_action_seq"].astype(int)
    trace_path = output_root / "meta" / "source_action_alignment.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    src_df[
        [
            "row_order",
            "index",
            "episode_index",
            "frame_index",
            "source",
            "is_backbone_fill",
            "source_is_hamer_valid",
            "image_abs_index",
            "image_frame_index",
            "source_dup_count",
            "is_replay_extra",
        ]
        + vec_info["names"]
    ].to_csv(trace_path, index=False)
    ep_summary = (
        src_df.groupby("episode_index")
        .agg(
            n=("index", "size"),
            src_min=("index", "min"),
            src_max=("index", "max"),
            hamer_valid=("source_is_hamer_valid", "sum"),
            replay_extra=("is_replay_extra", "sum"),
            backbone_fill=("is_backbone_fill", "sum"),
        )
        .reset_index()
    )
    print("[INFO] episode summary (aligned source index):")
    print(ep_summary.to_string(index=False))
    print(f"[INFO] alignment trace: {trace_path}")

    if image_video_path is None:
        video_files = sorted((Path(src_ds.root) / "videos").glob("**/*.mp4"))
        if len(video_files) != 1:
            raise RuntimeError(
                f"Expected exactly 1 source video for now, got {len(video_files)}. "
                "Please provide --hand-video-path explicitly."
            )
        video_path = video_files[0]
    else:
        video_path = Path(image_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"hand-video-path not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_path}")

    robot_cap = None
    robot_last_idx: int | None = None
    robot_last_rgb: np.ndarray | None = None
    robot_total_frames = 0
    if robot_video_path is not None:
        robot_cap = cv2.VideoCapture(str(robot_video_path))
        if not robot_cap.isOpened():
            raise RuntimeError(f"Failed to open robot video: {robot_video_path}")
        robot_total_frames = int(robot_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_indices = src_df["image_frame_index"].to_numpy(dtype=np.int64)
    selected_ptr = 0
    last_idx: int | None = None
    last_rgb: np.ndarray | None = None

    def _task_name(task_idx: int) -> str:
        return str(src_ds.meta.tasks.iloc[int(task_idx)].name)

    total = int(len(src_df))
    pbar = tqdm(total=total, desc="Rebuild selected frames", unit="frame", dynamic_ncols=True)

    try:
        current_episode = None
        while selected_ptr < len(selected_indices):
            if INTERRUPTED:
                raise KeyboardInterrupt("Interrupted by user")
            target_idx = int(selected_indices[selected_ptr])
            if last_idx is None or target_idx != last_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read source video at frame {target_idx}.")
                last_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                last_idx = target_idx

            row = src_df.iloc[selected_ptr]
            episode_idx = int(row["episode_index"])
            if current_episode is None:
                current_episode = episode_idx
            elif episode_idx != current_episode:
                new_ds.save_episode(parallel_encoding=False)
                current_episode = episode_idx

            assert last_rgb is not None
            vec = row[vec_info["names"]].to_numpy(dtype=np.float32)
            frame = {
                "observation.images.d435": last_rgb,
                "action": vec,
                "observation.state": vec,
                "action_valid": np.array([1.0], dtype=np.float32),
                "task": _task_name(int(row["task_index"])),
            }
            if robot_cap is not None:
                action_seq = int(row["robot_action_seq"]) if "robot_action_seq" in row.index else int(selected_ptr)
                if robot_total_frames <= 0:
                    raise RuntimeError("Robot video has zero frames.")
                action_seq = int(np.clip(action_seq, 0, robot_total_frames - 1))
                if robot_last_idx is None or action_seq != robot_last_idx:
                    robot_cap.set(cv2.CAP_PROP_POS_FRAMES, action_seq)
                    ok_robot, robot_bgr = robot_cap.read()
                    if not ok_robot:
                        raise RuntimeError(f"Failed to read robot video at frame {action_seq}.")
                    robot_last_rgb = cv2.cvtColor(robot_bgr, cv2.COLOR_BGR2RGB)
                    robot_last_idx = action_seq
                assert robot_last_rgb is not None
                frame["observation.images.robot"] = robot_last_rgb
            new_ds.add_frame(frame)
            pbar.update(1)
            selected_ptr += 1

        if current_episode is not None and new_ds.episode_buffer is not None and new_ds.episode_buffer["size"] > 0:
            new_ds.save_episode(parallel_encoding=False)
    finally:
        cap.release()
        if robot_cap is not None:
            robot_cap.release()
        pbar.close()
        new_ds.finalize()
        images_dir = Path(new_ds.root) / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)

    return new_ds


def main() -> None:
    global INTERRUPTED

    def _sigint_handler(_signum, _frame):
        # Mark interruption so long loops can stop at the next safe point.
        global INTERRUPTED
        INTERRUPTED = True
        raise KeyboardInterrupt

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    parser = argparse.ArgumentParser(
        description="Create ACT-trainable LeRobot dataset by adding action + observation.state from replay CSV."
    )
    parser.add_argument("--dataset-root", required=True, help="Existing LeRobot dataset root, e.g. data/raw/5.7")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Replay CSV path with mid_base/tool_axis/gripper. Required when --action-csv-path is not set.",
    )
    parser.add_argument(
        "--action-csv-path",
        default=None,
        help=(
            "Optional CSV exported from replay actual send_action values. "
            "When set, action/state are loaded directly from this file and CSV->IK conversion is skipped."
        ),
    )
    parser.add_argument("--output-root", required=True, help="Output dataset root, e.g. data/act_ready/5.7")
    parser.add_argument("--repo-id", required=True, help="New dataset repo_id metadata, e.g. rita/5.7_act")
    parser.add_argument(
        "--align-by",
        choices=["index", "frame_index"],
        default="index",
        help="How to align rows between parquet and csv. Default=index.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["qpos", "ee"],
        default="qpos",
        help="action/state representation. Default qpos (official-like). ee is optional.",
    )
    parser.add_argument("--target-offset-x", type=float, default=0.0, help="Apply same replay target offset on x")
    parser.add_argument("--target-offset-y", type=float, default=0.0, help="Apply same replay target offset on y")
    parser.add_argument("--target-offset-z", type=float, default=0.0, help="Apply same replay target offset on z")
    parser.add_argument(
        "--hamer-render-video",
        type=str,
        default=None,
        help="Optional HaMeR render mp4 path. If provided, replace output RGB video with it.",
    )
    parser.add_argument(
        "--allow-hamer-video-frame-mismatch",
        action="store_true",
        help="Allow replacing RGB even if HaMeR video frame count mismatches dataset frames.",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted dataset to Hugging Face Hub.")
    parser.add_argument("--private", action="store_true", help="Create private dataset repo when pushing to hub.")
    parser.add_argument(
        "--hf-tags",
        type=str,
        default="",
        help="Comma-separated tags for dataset card when pushing to hub.",
    )
    parser.add_argument(
        "--video-backend",
        choices=["pyav", "torchcodec"],
        default="pyav",
        help="Video decode backend when reading source dataset frames. Default=pyav.",
    )
    parser.add_argument(
        "--pkl-path",
        default=None,
        help="HaMeR output PKL path for valid-frame alignment (used in --action-csv-path).",
    )
    parser.add_argument(
        "--hand-video-path",
        default=None,
        help="Video path used for hand-view image frames in action-csv mode (recommended: HaMeR processed render video).",
    )
    parser.add_argument(
        "--robot-video-path",
        default=None,
        help="Optional robot replay video path aligned with sent_action sequence; saved as observation.images.robot.",
    )
    args = parser.parse_args()

    try:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else None
        output_root = Path(args.output_root).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"dataset-root not found: {dataset_root}")
        if not args.action_csv_path:
            if csv_path is None:
                raise ValueError("--csv-path is required when --action-csv-path is not provided.")
            if not csv_path.exists():
                raise FileNotFoundError(f"csv-path not found: {csv_path}")

        from lerobot.datasets.dataset_tools import add_features
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id=args.repo_id, root=dataset_root, video_backend=args.video_backend)
        if args.action_csv_path:
            csv_df = None
            raw_valid_mask = np.ones((ds.num_frames,), dtype=np.float32)
        else:
            assert csv_path is not None
            csv_df, raw_valid_mask = _prepare_csv(csv_path)
            if args.align_by == "index" and len(csv_df) != ds.num_frames:
                raise ValueError(f"CSV rows ({len(csv_df)}) != dataset num_frames ({ds.num_frames}) for align-by=index.")

        if args.action_csv_path:
            action_csv_path = Path(args.action_csv_path).expanduser().resolve()
            if not action_csv_path.exists():
                raise FileNotFoundError(f"action-csv-path not found: {action_csv_path}")
            if not args.pkl_path:
                raise ValueError("--pkl-path is required when --action-csv-path is provided.")
            pkl_path = Path(args.pkl_path).expanduser().resolve()
            if not pkl_path.exists():
                raise FileNotFoundError(f"pkl-path not found: {pkl_path}")
            action_df = pd.read_csv(action_csv_path)
            source_indices, vectors = _resolve_action_csv_indices(action_df, int(ds.num_frames))
            valid_abs_indices = _load_pkl_valid_indices(pkl_path, int(ds.num_frames))
            # action-csv mode: rebuild dataset using only selected source frames.
            new_ds = _build_dataset_from_selected_frames(
                src_ds=ds,
                source_indices=source_indices,
                action_vectors=vectors,
                output_root=output_root,
                repo_id=args.repo_id,
                valid_abs_indices=valid_abs_indices,
                image_video_path=Path(args.hand_video_path).expanduser().resolve() if args.hand_video_path else None,
                robot_video_path=Path(args.robot_video_path).expanduser().resolve() if args.robot_video_path else None,
            )
            _patch_stats_for_added_features(Path(new_ds.root))
            if args.hamer_render_video:
                _replace_rgb_video_with_hamer_render(
                    output_root=Path(new_ds.root),
                    hamer_render_video=Path(args.hamer_render_video).expanduser().resolve(),
                    expected_frames=int(new_ds.num_frames),
                    allow_mismatch=bool(args.allow_hamer_video_frame_mismatch),
                )
            if args.push_to_hub:
                tags = [t.strip() for t in args.hf_tags.split(",") if t.strip()]
                print(
                    f"[INFO] pushing dataset to hub: repo_id={new_ds.repo_id}, "
                    f"private={bool(args.private)}, tags={tags}"
                )
                new_ds.push_to_hub(tags=tags or None, private=bool(args.private))
                print(f"[INFO] dataset pushed: https://huggingface.co/datasets/{new_ds.repo_id}")

            print(f"[INFO] source dataset: {dataset_root}")
            print(f"[INFO] csv: (unused, action-csv mode)")
            print(f"[INFO] action-csv: {action_csv_path}")
            print(f"[INFO] pkl-path: {pkl_path}")
            if args.hand_video_path:
                print(f"[INFO] hand-video-path: {Path(args.hand_video_path).expanduser().resolve()}")
            if args.robot_video_path:
                print(f"[INFO] robot-video-path: {Path(args.robot_video_path).expanduser().resolve()}")
            print(f"[INFO] output dataset: {new_ds.root}")
            print(f"[INFO] output repo_id: {new_ds.repo_id}")
            print(f"[INFO] total frames: {new_ds.num_frames}")
            print(f"[INFO] selected source frames: {len(source_indices)}")
            return
        else:
            assert csv_df is not None
            if args.action_mode == "qpos":
                vectors, ik_valid = _make_qpos_vectors(
                    csv_df,
                    target_offset_x=float(args.target_offset_x),
                    target_offset_y=float(args.target_offset_y),
                    target_offset_z=float(args.target_offset_z),
                )
                vectors = _convert_qpos_to_replay_real_degrees(vectors)
                names = [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ]
            else:
                vectors = _make_ee_vectors(
                    csv_df,
                    target_offset_x=float(args.target_offset_x),
                    target_offset_y=float(args.target_offset_y),
                    target_offset_z=float(args.target_offset_z),
                )
                ik_valid = np.ones((len(csv_df),), dtype=np.float32)
                names = ["ee.x", "ee.y", "ee.z", "ee.axis_x", "ee.axis_y", "ee.axis_z", "ee.gripper"]

        action_valid = (raw_valid_mask * ik_valid).astype(np.float32)

        if args.align_by == "index":
            mat = vectors
            valid_vec = action_valid.reshape(-1, 1)

            def row_to_action(row_dict, _ep_idx, _frame_in_ep):
                return mat[int(row_dict["index"])]

            def row_to_valid(row_dict, _ep_idx, _frame_in_ep):
                return valid_vec[int(row_dict["index"])]
        else:
            grouped_idx = {}
            grouped_valid = {}
            for _, row in csv_df.iterrows():
                fi = int(row["frame_index"])
                if fi in grouped_idx:
                    continue
                grouped_idx[fi] = vectors[_]
                grouped_valid[fi] = np.array([action_valid[_]], dtype=np.float32)

            def row_to_action(_row_dict, _ep_idx, frame_in_ep):
                fi = int(frame_in_ep)
                if fi not in grouped_idx:
                    raise KeyError(f"frame_index={fi} not found in csv.")
                return grouped_idx[fi]

            def row_to_valid(_row_dict, _ep_idx, frame_in_ep):
                fi = int(frame_in_ep)
                if fi not in grouped_valid:
                    raise KeyError(f"frame_index={fi} not found in csv.")
                return grouped_valid[fi]

        vec_info = {"dtype": "float32", "shape": (6,), "names": names}
        valid_info = {"dtype": "float32", "shape": (1,), "names": ["is_valid"]}

        if INTERRUPTED:
            raise KeyboardInterrupt("Interrupted by user before add_features")

        new_ds = add_features(
            dataset=ds,
            features={
                "action": (row_to_action, vec_info),
                "observation.state": (row_to_action, vec_info),
                "action_valid": (row_to_valid, valid_info),
            },
            output_dir=output_root,
            repo_id=args.repo_id,
        )
        _patch_stats_for_added_features(Path(new_ds.root))

        if args.hamer_render_video:
            _replace_rgb_video_with_hamer_render(
                output_root=Path(new_ds.root),
                hamer_render_video=Path(args.hamer_render_video).expanduser().resolve(),
                expected_frames=int(new_ds.num_frames),
                allow_mismatch=bool(args.allow_hamer_video_frame_mismatch),
            )

        if args.push_to_hub:
            tags = [t.strip() for t in args.hf_tags.split(",") if t.strip()]
            print(f"[INFO] pushing dataset to hub: repo_id={new_ds.repo_id}, private={bool(args.private)}, tags={tags}")
            new_ds.push_to_hub(tags=tags or None, private=bool(args.private))
            print(f"[INFO] dataset pushed: https://huggingface.co/datasets/{new_ds.repo_id}")

        print(f"[INFO] source dataset: {dataset_root}")
        print(f"[INFO] csv: {csv_path if csv_path else '(unused, action-csv mode)'}")
        print(f"[INFO] output dataset: {new_ds.root}")
        print(f"[INFO] output repo_id: {new_ds.repo_id}")
        print(f"[INFO] total frames: {new_ds.num_frames}")
        print(f"[INFO] action-mode: {args.action_mode}")
        print(
            f"[INFO] target offsets applied: "
            f"dx={float(args.target_offset_x):+.4f}, dy={float(args.target_offset_y):+.4f}, dz={float(args.target_offset_z):+.4f}"
        )
        print(
            f"[INFO] action_valid ratio: {float(action_valid.mean()):.3f} "
            f"(valid={int(action_valid.sum())} / total={len(action_valid)})"
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C, conversion stopped.", file=sys.stderr)
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()
