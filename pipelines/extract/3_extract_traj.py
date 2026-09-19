#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm as _real_tqdm
except Exception:
    _real_tqdm = None

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from common.config import (
    DEFAULT_EXTRACT_FPS_FALLBACK,
    get_default_camera_to_base_path,
)

DEFAULT_PLANE_KEYPOINT_INDICES = (2, 3, 4, 5, 6, 7, 8)
KP02 = 2
KP05 = 5
KP04 = 4
KP08 = 8


def _tqdm(iterable, **kwargs):
    if _real_tqdm is None:
        return iterable
    return _real_tqdm(iterable, **kwargs)


def load_pkl(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected top-level dict, got {type(obj)}")
    return obj


def load_intrinsics_json(path: str):
    """
    读取 depth 相机内参 JSON
    期望格式示例:
    {
      "intrinsics": {
        "fx": 615.0,
        "fy": 615.0,
        "ppx": 320.0,
        "ppy": 240.0
      },
      "depth_scale_m_per_unit": 0.001
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intr = data["intrinsics"]
    depth_scale = float(data["depth_scale_m_per_unit"])
    for k in ["fx", "fy", "ppx", "ppy"]:
        if k not in intr:
            raise KeyError(f"Missing intrinsics key: {k}")
    return intr, depth_scale


def load_depth_npz(path: str) -> np.ndarray:
    """
    读取 depth.npz，要求里面有 depth_mm
    shape: (T, H, W)
    """
    obj = np.load(path)
    if "depth_mm" not in obj:
        raise KeyError(f"'depth_mm' not found in {path}. Keys: {list(obj.keys())}")
    return obj["depth_mm"]


def load_depth_episode_index(path: str, total_depth_frames: int) -> np.ndarray:
    """
    Load depth_index.json and build depth-frame -> episode_index lookup.
    Returns int64 array of shape [total_depth_frames], default -1.
    """
    episode_by_depth = np.full((total_depth_frames,), -1, dtype=np.int64)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    for item in episodes:
        try:
            episode_index = int(item["episode_index"])
            start = int(item["start"])
            end = int(item["end"])
        except Exception:
            continue
        if end <= start:
            continue
        start = max(0, start)
        end = min(total_depth_frames, end)
        if end > start:
            episode_by_depth[start:end] = episode_index
    return episode_by_depth


def load_episode_timestamps(path: str) -> np.ndarray:
    """读取 timestamps.npy（秒）"""
    timestamps = np.load(path)
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if timestamps.size == 0:
        raise ValueError(f"Empty timestamps in {path}")
    return timestamps


def load_frame_meta_timestamps(path: str) -> np.ndarray:
    """从 frame_meta.jsonl 读取 timestamp_s（秒）"""
    values: list[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "timestamp_s" in payload:
                values.append(float(payload["timestamp_s"]))
    if not values:
        raise ValueError(f"No timestamp_s found in {path}")
    return np.asarray(values, dtype=float)


def load_cam2base(path: Path) -> np.ndarray:
    # Keep exactly the same key-handling logic as calibration/2_click_to_base_verify.py
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "T_cam2base" in payload:
        matrix = np.asarray(payload["T_cam2base"], dtype=float)
    elif "T_base_camera" in payload:
        matrix = np.asarray(payload["T_base_camera"], dtype=float)
    else:
        raise KeyError("Transform JSON must contain 'T_cam2base' or 'T_base_camera'.")
    if matrix.shape != (4, 4):
        raise ValueError(f"T_cam2base must be (4,4), got {matrix.shape}")
    return matrix


def rotmat_to_rpy_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """
    ZYX 欧拉角
    返回 roll, pitch, yaw，单位 rad
    """
    R = np.asarray(R, dtype=float)
    if R.shape == (1, 3, 3):
        R = R[0]
    if R.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (3,3) or (1,3,3), got {R.shape}")

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return float(roll), float(pitch), float(yaw)


def extract_right_hand_indices(frame_data: dict[str, Any]) -> list[int]:
    mano = frame_data.get("mano", [])
    out = []
    for i, hand in enumerate(mano):
        try:
            if int(hand.get("is_right", 0)) == 1:
                out.append(i)
        except Exception:
            continue
    return out


def select_hand_index(frame_data: dict[str, Any], hand: str = "right") -> int | None:
    mano = frame_data.get("mano", [])
    if len(mano) == 0:
        return None

    if hand == "right":
        idxs = extract_right_hand_indices(frame_data)
        return None if len(idxs) == 0 else idxs[0]

    if hand == "left":
        for i, m in enumerate(mano):
            try:
                if int(m.get("is_right", 0)) == 0:
                    return i
            except Exception:
                continue
        return None

    # hand == "first"
    return 0


def select_hand_index_with_track(
    frame_data: dict[str, Any],
    hand: str,
    preferred_track_id: int | None,
) -> int | None:
    """Select hand index with optional tracked id preference."""
    cam_trans = frame_data.get("cam_trans", [])
    tracked_ids = frame_data.get("tracked_ids", [])
    hand_count = len(cam_trans) if hasattr(cam_trans, "__len__") else 0
    if hand_count <= 0:
        return None

    # 1) If preferred track id exists in current frame, use it first.
    if preferred_track_id is not None and hasattr(tracked_ids, "__len__"):
        for hand_index in range(min(hand_count, len(tracked_ids))):
            try:
                if int(tracked_ids[hand_index]) == int(preferred_track_id):
                    return hand_index
            except Exception:
                continue

    # 2) Fallback to original selection logic (right/left/first).
    return select_hand_index(frame_data, hand=hand)


def safe_list_get(lst: Any, idx: int, default=None):
    try:
        return lst[idx]
    except Exception:
        return default


def deproject_pixel_like_verify(u: float, v: float, depth_m: float, intr: dict):
    """
    Match calibration/2_click_to_base_verify.py: rs2_deproject_pixel_to_point behavior.
    """
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    ppx = float(intr["ppx"])
    ppy = float(intr["ppy"])
    x = (float(u) - ppx) * float(depth_m) / fx
    y = (float(v) - ppy) * float(depth_m) / fy
    z = float(depth_m)
    return np.array([x, y, z], dtype=float)


def get_depth_median_at_pixel(depth_img: np.ndarray, u: int, v: int, patch_radius: int = 1) -> float:
    # Keep same neighborhood-median style as calibration/2_click_to_base_verify.py
    h, w = depth_img.shape
    uc = int(u)
    vc = int(v)
    values = []
    for dy in range(-patch_radius, patch_radius + 1):
        for dx in range(-patch_radius, patch_radius + 1):
            uu = uc + dx
            vv = vc + dy
            if uu < 0 or vv < 0 or uu >= w or vv >= h:
                continue
            depth_value = float(depth_img[vv, uu])
            if depth_value > 0:
                values.append(depth_value)
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def keypoint_2d_from_frame(frame_data: dict[str, Any], hand_idx: int, kp_idx: int):
    """
    从 extra_data 中取某个关键点 2D
    期望 shape: (21, 3)，第三列通常是 conf
    """
    kps = safe_list_get(frame_data.get("extra_data", []), hand_idx, default=None)
    if kps is None:
        return None

    kps = np.asarray(kps, dtype=float)
    if kps.ndim != 2 or kps.shape[0] <= kp_idx or kps.shape[1] < 2:
        return None

    u = kps[kp_idx, 0]
    v = kps[kp_idx, 1]
    conf = kps[kp_idx, 2] if kps.shape[1] >= 3 else math.nan

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return float(u), float(v), float(conf)


def stabilize_normal(
    current_normal: np.ndarray,
    previous_normal: np.ndarray | None,
    ema_alpha: float,
    max_step_deg: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(current_normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    if previous_normal is not None:
        previous_normal = np.asarray(previous_normal, dtype=float)
        previous_normal = previous_normal / (np.linalg.norm(previous_normal) + 1e-12)
        if float(np.dot(normal, previous_normal)) < 0.0:
            normal = -normal
        # Limit per-frame angular change to suppress outlier spikes.
        dot_val = float(np.clip(np.dot(previous_normal, normal), -1.0, 1.0))
        angle = math.degrees(math.acos(dot_val))
        if angle > max_step_deg:
            t = max_step_deg / max(angle, 1e-12)
            normal = (1.0 - t) * previous_normal + t * normal
            normal = normal / (np.linalg.norm(normal) + 1e-12)
        if ema_alpha > 0.0:
            normal = (1.0 - ema_alpha) * previous_normal + ema_alpha * normal
            normal = normal / (np.linalg.norm(normal) + 1e-12)

    return normal, normal.copy()


def robust_plane_normal_ransac(
    points_cam: np.ndarray,
    ransac_iters: int = 64,
    inlier_threshold_m: float = 0.008,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Robustly estimate a plane normal from 3D points using lightweight RANSAC.
    Returns (unit_normal, inlier_mask).
    """
    points = np.asarray(points_cam, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        return None

    n = points.shape[0]
    best_count = -1
    best_mask = None
    best_normal = None

    rng = np.random.default_rng(42)
    for _ in range(ransac_iters):
        sample = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[sample]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        distances = np.abs((points - p0) @ normal)
        mask = distances <= inlier_threshold_m
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            best_normal = normal

    if best_normal is None or best_mask is None:
        return None

    inliers = points[best_mask]
    if inliers.shape[0] < 3:
        inliers = points
        best_mask = np.ones(n, dtype=bool)

    centered = inliers - np.mean(inliers, axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    refined_normal = right_vectors[-1]
    refined_norm = np.linalg.norm(refined_normal)
    if refined_norm < 1e-9:
        return None
    refined_normal = refined_normal / refined_norm
    return refined_normal, best_mask




def build_orientation_from_seven_keypoints(
    frame_data: dict[str, Any],
    hand_idx: int,
    depth_img: np.ndarray,
    intr: dict,
    depth_scale: float,
    patch_radius: int,
    plane_keypoint_indices: tuple[int, ...] = DEFAULT_PLANE_KEYPOINT_INDICES,
    previous_plane_normal: np.ndarray | None = None,
    normal_ema_alpha: float = 0.35,
    normal_max_step_deg: float = 12.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    points_cam = []
    for keypoint_index in plane_keypoint_indices:
        keypoint = keypoint_2d_from_frame(frame_data, hand_idx, kp_idx=keypoint_index)
        if keypoint is None:
            return None
        key_u, key_v, _ = keypoint
        depth_raw = get_depth_median_at_pixel(
            depth_img=depth_img,
            u=int(round(key_u)),
            v=int(round(key_v)),
            patch_radius=patch_radius,
        )
        if depth_raw <= 0:
            return None
        depth_m = float(depth_raw) * float(depth_scale)
        point_cam = deproject_pixel_like_verify(key_u, key_v, depth_m, intr)
        points_cam.append(point_cam)
    points_cam = np.asarray(points_cam, dtype=float)

    robust = robust_plane_normal_ransac(points_cam)
    if robust is None:
        return None
    plane_normal, _ = robust
    if plane_normal[2] < 0:
        plane_normal = -plane_normal
    plane_normal, updated_previous = stabilize_normal(
        current_normal=plane_normal,
        previous_normal=previous_plane_normal,
        ema_alpha=normal_ema_alpha,
        max_step_deg=normal_max_step_deg,
    )

    # Keep rotation build for compatibility (yaw can be ignored downstream).
    camera_x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = camera_x_axis - np.dot(camera_x_axis, plane_normal) * plane_normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        camera_y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = camera_y_axis - np.dot(camera_y_axis, plane_normal) * plane_normal
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            return None
    x_axis = x_axis / x_norm

    y_axis = np.cross(plane_normal, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return None
    y_axis = y_axis / y_norm

    z_axis = plane_normal

    rotation_cam = np.column_stack([x_axis, y_axis, z_axis])
    return rotation_cam, updated_previous


def make_row_base(frame_index: int, frame_path: str, timestamp_s: float, relative_t: float):
    return {
        "frame_index": frame_index,
        "episode_index": -1,
        "frame_path": frame_path,
        "frame_id": -1,
        "segment_id": -1,
        "shot_id": -1,
        "track_id": -1,
        "timestamp_s": float(timestamp_s),
        "t": float(relative_t),
        "hand_found": 0,

        "kp02_u": math.nan,
        "kp02_v": math.nan,
        "kp02_conf": math.nan,
        "kp05_u": math.nan,
        "kp05_v": math.nan,
        "kp05_conf": math.nan,

        "kp02_cam_x": math.nan,
        "kp02_cam_y": math.nan,
        "kp02_cam_z": math.nan,
        "kp05_cam_x": math.nan,
        "kp05_cam_y": math.nan,
        "kp05_cam_z": math.nan,

        "mid_cam_x": math.nan,
        "mid_cam_y": math.nan,
        "mid_cam_z": math.nan,

        "mid_base_x": math.nan,
        "mid_base_y": math.nan,
        "mid_base_z": math.nan,

        "cam_roll": math.nan,
        "cam_pitch": math.nan,
        "cam_yaw": math.nan,

        "tool_axis_x": math.nan,
        "tool_axis_y": math.nan,
        "tool_axis_z": math.nan,
        "pinch_distance_m": math.nan,
        "gripper_cmd": math.nan,
        "pinch_source": "",
    }


def parse_frame_id_from_path(frame_path: str) -> int:
    """Parse frame id from .../000123.jpg. Return -1 if unavailable."""
    stem = Path(frame_path).stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return -1


def infer_preferred_track_id(data: dict[str, Any]) -> int | None:
    """Choose the most frequent tracked id across valid frames."""
    counts: dict[int, int] = {}
    for frame_data in data.values():
        if not isinstance(frame_data, dict):
            continue
        cam_trans = frame_data.get("cam_trans", [])
        tracked_ids = frame_data.get("tracked_ids", [])
        if not hasattr(cam_trans, "__len__") or not hasattr(tracked_ids, "__len__"):
            continue
        hand_count = min(len(cam_trans), len(tracked_ids))
        for hand_index in range(hand_count):
            try:
                track_id = int(tracked_ids[hand_index])
            except Exception:
                continue
            counts[track_id] = counts.get(track_id, 0) + 1

    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def assign_segment_ids(rows: list[dict[str, Any]], max_missing_gap: int) -> None:
    """
    Assign contiguous segment ids based on:
    - hand_found transitions
    - shot boundary changes
    - missing gap threshold
    """
    current_segment_id = -1
    missing_run = 0
    previous_shot_id = None

    for row in rows:
        shot_id = int(row.get("shot_id", -1))
        hand_found = int(row.get("hand_found", 0)) == 1

        shot_changed = previous_shot_id is not None and shot_id != previous_shot_id
        previous_shot_id = shot_id

        if hand_found:
            if current_segment_id < 0:
                current_segment_id = 0
            elif shot_changed or missing_run > max_missing_gap:
                current_segment_id += 1
            missing_run = 0
            row["segment_id"] = current_segment_id
        else:
            missing_run += 1
            row["segment_id"] = -1


def build_rows(
    data: dict[str, Any],
    depth_stack: np.ndarray,
    intr: dict,
    depth_scale: float,
    episode_by_depth: np.ndarray,
    T_cam2base: np.ndarray,
    frame_timestamps_s: np.ndarray,
    fallback_fps: float,
    hand: str,
    preferred_track_id: int | None,
    patch_radius: int,
    keep_missing: bool,
    max_missing_gap: int,
    plane_normal_ema_alpha: float,
    plane_normal_max_step_deg: float,
    pinch_close_m: float,
    pinch_open_m: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame_paths = sorted(data.keys())
    rows = []
    alignment_stats = {
        "total_frames": len(frame_paths),
        "depth_by_frame_id": 0,
        "depth_by_index_fallback": 0,
        "timestamp_by_frame_id": 0,
        "timestamp_by_index_or_fallback": 0,
    }

    R_cam2base = T_cam2base[:3, :3]
    t_cam2base = T_cam2base[:3, 3]

    total_depth_frames = len(depth_stack)
    total_timestamps = len(frame_timestamps_s)
    base_timestamp = float(frame_timestamps_s[0]) if total_timestamps > 0 else 0.0

    previous_plane_normal: np.ndarray | None = None
    previous_base_normal: np.ndarray | None = None
    iterator = _tqdm(
        enumerate(frame_paths),
        total=len(frame_paths),
        desc="extract traj rows",
        unit="frame",
    )
    for frame_index, frame_path in iterator:
        frame_data = data[frame_path]
        frame_id = parse_frame_id_from_path(frame_path)
        # Dyn-HaMR frame file names are 1-based (000001.jpg ...), while depth/timestamp arrays are 0-based.
        frame_id_zero_based = frame_id - 1 if frame_id > 0 else frame_id
        if 0 <= frame_id_zero_based < total_timestamps:
            timestamp_s = float(frame_timestamps_s[frame_id_zero_based])
            alignment_stats["timestamp_by_frame_id"] += 1
        elif frame_index < total_timestamps:
            timestamp_s = float(frame_timestamps_s[frame_index])
            alignment_stats["timestamp_by_index_or_fallback"] += 1
        else:
            timestamp_s = base_timestamp + frame_index / float(fallback_fps)
            alignment_stats["timestamp_by_index_or_fallback"] += 1
        row = make_row_base(
            frame_index=frame_index,
            frame_path=frame_path,
            timestamp_s=timestamp_s,
            relative_t=timestamp_s - base_timestamp,
        )
        row["frame_id"] = frame_id
        try:
            row["shot_id"] = int(frame_data.get("shot", -1))
        except Exception:
            row["shot_id"] = -1

        depth_index = frame_id_zero_based if 0 <= frame_id_zero_based < total_depth_frames else frame_index
        if depth_index >= total_depth_frames:
            if keep_missing:
                rows.append(row)
            continue
        if 0 <= frame_id_zero_based < total_depth_frames:
            alignment_stats["depth_by_frame_id"] += 1
        else:
            alignment_stats["depth_by_index_fallback"] += 1
        if 0 <= depth_index < len(episode_by_depth):
            row["episode_index"] = int(episode_by_depth[depth_index])

        hand_idx = select_hand_index_with_track(
            frame_data=frame_data,
            hand=hand,
            preferred_track_id=preferred_track_id,
        )
        if hand_idx is None:
            if keep_missing:
                rows.append(row)
            continue

        kp02 = keypoint_2d_from_frame(frame_data, hand_idx, kp_idx=2)
        kp05 = keypoint_2d_from_frame(frame_data, hand_idx, kp_idx=5)
        p2_cam: np.ndarray | None = None
        p5_cam: np.ndarray | None = None

        if kp02 is not None:
            u2, v2, c2 = kp02
            row["kp02_u"] = u2
            row["kp02_v"] = v2
            row["kp02_conf"] = c2
        if kp05 is not None:
            u5, v5, c5 = kp05
            row["kp05_u"] = u5
            row["kp05_v"] = v5
            row["kp05_conf"] = c5

        if kp02 is None or kp05 is None:
            if keep_missing:
                rows.append(row)
            continue
        depth_img = depth_stack[depth_index]
        d2_raw = get_depth_median_at_pixel(
            depth_img=depth_img,
            u=int(round(row["kp02_u"])),
            v=int(round(row["kp02_v"])),
            patch_radius=patch_radius,
        )
        d5_raw = get_depth_median_at_pixel(
            depth_img=depth_img,
            u=int(round(row["kp05_u"])),
            v=int(round(row["kp05_v"])),
            patch_radius=patch_radius,
        )
        if d2_raw <= 0 or d5_raw <= 0:
            if keep_missing:
                rows.append(row)
            continue
        d2_m = float(d2_raw) * float(depth_scale)
        d5_m = float(d5_raw) * float(depth_scale)
        p2_cam = deproject_pixel_like_verify(row["kp02_u"], row["kp02_v"], d2_m, intr)
        p5_cam = deproject_pixel_like_verify(row["kp05_u"], row["kp05_v"], d5_m, intr)
        pmid_cam = 0.5 * (p2_cam + p5_cam)

        # Pinch is defined only by fingertips (thumb tip kp04 <-> index tip kp08),
        # using XY-plane distance in camera frame (ignore Z as requested).
        pinch_distance_m = math.nan
        pinch_source = ""
        kp04 = keypoint_2d_from_frame(frame_data, hand_idx, kp_idx=KP04)
        kp08 = keypoint_2d_from_frame(frame_data, hand_idx, kp_idx=KP08)
        if kp04 is not None and kp08 is not None:
            u4, v4, _ = kp04
            u8, v8, _ = kp08
            d4_raw = get_depth_median_at_pixel(
                depth_img=depth_img,
                u=int(round(u4)),
                v=int(round(v4)),
                patch_radius=patch_radius,
            )
            d8_raw = get_depth_median_at_pixel(
                depth_img=depth_img,
                u=int(round(u8)),
                v=int(round(v8)),
                patch_radius=patch_radius,
            )
            if d4_raw > 0 and d8_raw > 0:
                d4_m = float(d4_raw) * float(depth_scale)
                d8_m = float(d8_raw) * float(depth_scale)
                p4_cam = deproject_pixel_like_verify(u4, v4, d4_m, intr)
                p8_cam = deproject_pixel_like_verify(u8, v8, d8_m, intr)
                pinch_distance_m = float(np.linalg.norm((p4_cam - p8_cam)[:2]))
                pinch_source = "kp04_kp08"

        pmid_base = R_cam2base @ pmid_cam + t_cam2base

        row["hand_found"] = 1
        try:
            tracked_ids = frame_data.get("tracked_ids", [])
            if hasattr(tracked_ids, "__len__") and hand_idx < len(tracked_ids):
                row["track_id"] = int(tracked_ids[hand_idx])
        except Exception:
            row["track_id"] = -1

        if p2_cam is not None:
            row["kp02_cam_x"] = float(p2_cam[0])
            row["kp02_cam_y"] = float(p2_cam[1])
            row["kp02_cam_z"] = float(p2_cam[2])
        if p5_cam is not None:
            row["kp05_cam_x"] = float(p5_cam[0])
            row["kp05_cam_y"] = float(p5_cam[1])
            row["kp05_cam_z"] = float(p5_cam[2])

        row["mid_cam_x"] = float(pmid_cam[0])
        row["mid_cam_y"] = float(pmid_cam[1])
        row["mid_cam_z"] = float(pmid_cam[2])
        row["pinch_distance_m"] = pinch_distance_m
        row["pinch_source"] = pinch_source
        if np.isfinite(pinch_distance_m) and pinch_open_m > pinch_close_m:
            ratio = (pinch_distance_m - pinch_close_m) / (pinch_open_m - pinch_close_m)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            # Unified gripper semantic for this project:
            #   0.0 -> fully closed
            #   1.0 -> fully open
            row["gripper_cmd"] = ratio

        row["mid_base_x"] = float(pmid_base[0])
        row["mid_base_y"] = float(pmid_base[1])
        row["mid_base_z"] = float(pmid_base[2])

        try:
            rotation_cam = None
            depth_img = depth_stack[depth_index]
            result = build_orientation_from_seven_keypoints(
                frame_data=frame_data,
                hand_idx=hand_idx,
                depth_img=depth_img,
                intr=intr,
                depth_scale=float(depth_scale),
                patch_radius=patch_radius,
                previous_plane_normal=previous_plane_normal,
                normal_ema_alpha=plane_normal_ema_alpha,
                normal_max_step_deg=plane_normal_max_step_deg,
            )
            if result is not None:
                rotation_cam, previous_plane_normal = result
            if rotation_cam is None:
                mano_item = safe_list_get(frame_data.get("mano", []), hand_idx, default=None)
                if mano_item is not None and "global_orient" in mano_item:
                    rotation_cam = np.asarray(mano_item["global_orient"], dtype=float)
                    if rotation_cam.shape == (1, 3, 3):
                        rotation_cam = rotation_cam[0]
            if rotation_cam is not None:
                rotation_base = R_cam2base @ rotation_cam
                cam_roll, cam_pitch, cam_yaw = rotmat_to_rpy_zyx(rotation_cam)
                row["cam_roll"] = cam_roll
                row["cam_pitch"] = cam_pitch
                row["cam_yaw"] = cam_yaw

                # Tool axis = z-column of base-frame rotation (palm normal direction).
                # Stored directly as a unit vector — no Euler angle conversion needed.
                base_normal = rotation_base[:, 2]
                base_normal, previous_base_normal = stabilize_normal(
                    current_normal=base_normal,
                    previous_normal=previous_base_normal,
                    ema_alpha=plane_normal_ema_alpha,
                    max_step_deg=plane_normal_max_step_deg,
                )
                row["tool_axis_x"] = float(base_normal[0])
                row["tool_axis_y"] = float(base_normal[1])
                row["tool_axis_z"] = float(base_normal[2])
        except Exception:
            pass

        rows.append(row)

        # Fallback progress when tqdm is unavailable in current environment.
        if _real_tqdm is None:
            progressed = frame_index + 1
            if (progressed % 120 == 0) or (progressed == len(frame_paths)):
                print(f"[INFO] extract traj rows: {progressed}/{len(frame_paths)}")

    assign_segment_ids(rows=rows, max_missing_gap=max_missing_gap)
    return rows, alignment_stats


def write_csv(rows: list[dict[str, Any]], output_path: str):
    if len(rows) == 0:
        raise ValueError("No rows to write")
    output_parent = Path(output_path).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def postprocess_rows_for_training(
    rows: list[dict[str, Any]],
    smooth_window: int,
    gripper_min: float,
    gripper_max: float,
) -> None:
    if not rows:
        return

    # Dyn-HaMR can briefly lose the hand in the middle of an otherwise valid
    # sequence.  A missing hand must not be interpreted as a new gripper
    # measurement: preserve the last observed opening until the hand is found
    # again.  Leading missing rows intentionally remain NaN because no prior
    # gripper state exists to hold.
    last_gripper_cmd = math.nan
    for row in rows:
        try:
            gripper_cmd = float(row.get("gripper_cmd", math.nan))
        except (TypeError, ValueError):
            gripper_cmd = math.nan

        if np.isfinite(gripper_cmd):
            last_gripper_cmd = gripper_cmd
            continue

        if int(row.get("hand_found", 0)) == 0 and np.isfinite(last_gripper_cmd):
            row["gripper_cmd"] = float(last_gripper_cmd)
            row["pinch_source"] = "hold_last_missing_hand"

    df = pd.DataFrame(rows)
    core_cols = ["mid_base_x", "mid_base_y", "mid_base_z", "tool_axis_x", "tool_axis_y", "tool_axis_z", "gripper_cmd"]
    for col in core_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep a fully aligned sequence for training/replay.
    present_core = [c for c in core_cols if c in df.columns]
    if present_core:
        df[present_core] = df[present_core].interpolate(limit_direction="both").ffill().bfill()

    if smooth_window > 1 and present_core:
        if smooth_window % 2 == 0:
            smooth_window += 1
        for col in present_core:
            df[col] = df[col].rolling(window=smooth_window, center=True, min_periods=1).median()

    if "gripper_cmd" in df.columns:
        df["gripper_cmd"] = df["gripper_cmd"].clip(float(gripper_min), float(gripper_max))

    # Write back to original row dicts to keep existing CSV field order/logic.
    for i in range(len(rows)):
        for col in present_core:
            value = df.at[i, col]
            rows[i][col] = float(value) if pd.notna(value) else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pkl", required=True, help="Dyn-HaMR output pkl")
    ap.add_argument("--episode-dir", required=True, help="contains depth.npz and intrinsics.json")
    ap.add_argument(
        "--transform-json",
        default=str(get_default_camera_to_base_path()),
        help="contains T_cam2base; default uses calibration/outputs/cam2base_latest.json",
    )
    ap.add_argument("--output-csv", required=True)
    ap.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_EXTRACT_FPS_FALLBACK,
        help="fallback only when timestamps are unavailable",
    )
    ap.add_argument("--hand", choices=["right", "left", "first"], default="right")
    ap.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Prefer a fixed tracked id. Default: auto-infer dominant tracked id.",
    )
    ap.add_argument("--patch-radius", type=int, default=2)
    ap.add_argument(
        "--max-missing-gap",
        type=int,
        default=8,
        help="Segment split threshold for consecutive missing frames.",
    )
    ap.add_argument(
        "--plane-normal-ema-alpha",
        type=float,
        default=0.35,
        help="EMA alpha for seven-keypoint plane normal smoothing, range [0,1].",
    )
    ap.add_argument(
        "--plane-normal-max-step-deg",
        type=float,
        default=12.0,
        help="Max allowed normal direction change per frame (deg).",
    )
    ap.add_argument(
        "--pinch-close-m",
        type=float,
        default=0.000,
        help="Finger distance (meters) mapped to gripper close endpoint (0.0).",
    )
    ap.add_argument(
        "--pinch-open-m",
        type=float,
        default=0.138,
        help="Finger distance (meters) mapped to gripper open endpoint (1.0).",
    )
    ap.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Median smoothing window for mid_base/tool_axis/gripper before CSV output.",
    )
    ap.add_argument("--gripper-min", type=float, default=0.0, help="Clamp min for gripper_cmd.")
    ap.add_argument("--gripper-max", type=float, default=1.0, help="Clamp max for gripper_cmd.")
    args = ap.parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if not (0.0 <= args.plane_normal_ema_alpha <= 1.0):
        raise ValueError("--plane-normal-ema-alpha must be in [0,1]")
    if args.plane_normal_max_step_deg <= 0:
        raise ValueError("--plane-normal-max-step-deg must be > 0")
    if args.pinch_open_m <= args.pinch_close_m:
        raise ValueError("--pinch-open-m must be greater than --pinch-close-m")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be > 0")
    if args.gripper_max <= args.gripper_min:
        raise ValueError("--gripper-max must be greater than --gripper-min")

    episode_dir = Path(args.episode_dir)
    depth_path = episode_dir / "depth.npz"
    intr_path = episode_dir / "intrinsics.json"
    depth_index_path = episode_dir / "depth_index.json"
    timestamps_path = episode_dir / "timestamps.npy"
    frame_meta_path = episode_dir / "frame_meta.jsonl"

    data = load_pkl(args.input_pkl)
    if not depth_path.exists() or not intr_path.exists() or not depth_index_path.exists():
        raise FileNotFoundError(
            f"Required files not found under episode-dir: {depth_path} / {intr_path} / {depth_index_path}"
        )
    depth_stack = load_depth_npz(str(depth_path))
    intr, depth_scale = load_intrinsics_json(str(intr_path))
    episode_by_depth = load_depth_episode_index(str(depth_index_path), total_depth_frames=len(depth_stack))
    T_cam2base = load_cam2base(Path(args.transform_json).expanduser().resolve())
    timestamp_source = "fps_fallback"
    if timestamps_path.exists():
        frame_timestamps_s = load_episode_timestamps(str(timestamps_path))
        timestamp_source = "timestamps.npy"
    elif frame_meta_path.exists():
        frame_timestamps_s = load_frame_meta_timestamps(str(frame_meta_path))
        timestamp_source = "frame_meta.jsonl"
    else:
        frame_count = len(sorted(data.keys()))
        frame_timestamps_s = np.arange(frame_count, dtype=float) / float(args.fps)

    rows, alignment_stats = build_rows(
        data=data,
        depth_stack=depth_stack,
        intr=intr,
        depth_scale=depth_scale,
        episode_by_depth=episode_by_depth,
        T_cam2base=T_cam2base,
        frame_timestamps_s=frame_timestamps_s,
        fallback_fps=args.fps,
        hand=args.hand,
        preferred_track_id=args.track_id if args.track_id is not None else infer_preferred_track_id(data),
        patch_radius=args.patch_radius,
        keep_missing=True,
        max_missing_gap=args.max_missing_gap,
        plane_normal_ema_alpha=args.plane_normal_ema_alpha,
        plane_normal_max_step_deg=args.plane_normal_max_step_deg,
        pinch_close_m=args.pinch_close_m,
        pinch_open_m=args.pinch_open_m,
    )
    postprocess_rows_for_training(
        rows=rows,
        smooth_window=int(args.smooth_window),
        gripper_min=float(args.gripper_min),
        gripper_max=float(args.gripper_max),
    )

    write_csv(rows, args.output_csv)

    total = len(sorted(data.keys()))
    kept = len(rows)
    found = sum(int(r["hand_found"]) for r in rows)
    valid_segment_ids = sorted({int(row["segment_id"]) for row in rows if int(row["segment_id"]) >= 0})

    print(f"[INFO] Total frames in PKL: {total}")
    print(f"[INFO] Rows written: {kept}")
    print(f"[INFO] Frames with valid kp02-kp05 midpoint: {found}")
    print(f"[INFO] Valid segments: {len(valid_segment_ids)} -> {valid_segment_ids}")
    print(f"[INFO] Timestamp source: {timestamp_source}")
    total_frames = max(1, int(alignment_stats["total_frames"]))
    depth_frame_id_ratio = float(alignment_stats["depth_by_frame_id"]) / float(total_frames)
    ts_frame_id_ratio = float(alignment_stats["timestamp_by_frame_id"]) / float(total_frames)
    print(
        "[INFO] Alignment: "
        f"depth frame_id hit {alignment_stats['depth_by_frame_id']}/{total_frames} "
        f"({depth_frame_id_ratio:.1%}), "
        f"timestamp frame_id hit {alignment_stats['timestamp_by_frame_id']}/{total_frames} "
        f"({ts_frame_id_ratio:.1%})"
    )
    if depth_frame_id_ratio < 0.9:
        print("[WARN] Depth alignment by frame_id is low; check Dyn-HaMR frame sampling FPS vs record FPS.")
    if ts_frame_id_ratio < 0.9:
        print("[WARN] Timestamp alignment by frame_id is low; check timestamp source or extracted frame ids.")
    print(f"[INFO] Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
