#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from common.config import (
    DEFAULT_VISUALIZE_EXPORT_FPS,
    get_default_camera_to_base_path,
)


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]

KP02 = 2
KP05 = 5
KP04 = 4
KP08 = 8
DEFAULT_PINCH_CLOSE_M = 0.000
DEFAULT_PINCH_OPEN_M = 0.140
# Unified runtime semantic:
# gripper_cmd in [0, 1], where 0=closed and 1=open.
DEFAULT_GRIPPER_CMD_MIN = 0.0
DEFAULT_GRIPPER_CMD_MAX = 1.0
PLANE_KEYPOINT_INDICES = (2, 3, 4, 5, 6, 7, 8)
VIS_PATCH_RADIUS = 0


def load_pkl(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected top-level dict, got {type(obj)}")
    return obj


def load_intrinsics_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intr = data["intrinsics"]
    depth_scale = float(data["depth_scale_m_per_unit"])
    return intr, depth_scale


def load_depth_npz(path: str) -> np.ndarray:
    obj = np.load(path)
    if "depth_mm" not in obj:
        raise KeyError(f"'depth_mm' not found in {path}. Keys: {list(obj.keys())}")
    return obj["depth_mm"]


def load_transform_json(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "T_cam2base" in data:
        T = np.asarray(data["T_cam2base"], dtype=float)
    elif "T_base_camera" in data:
        T = np.asarray(data["T_base_camera"], dtype=float)
    else:
        raise KeyError("Transform JSON must contain 'T_cam2base' or 'T_base_camera'.")
    if T.shape != (4, 4):
        raise ValueError(f"T_cam2base must be (4,4), got {T.shape}")
    return T


class VideoFrameReader:
    """On-demand video frame reader with LRU cache to keep memory bounded."""

    def __init__(self, path: str, cache_size: int = 128):
        self.path = path
        self.cache_size = max(8, int(cache_size))
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.last_idx = -1

        first = self.get_frame(0)
        if first is None:
            raise RuntimeError(f"No frames loaded from: {path}")
        self.frame_shape = first.shape

    def _touch_cache(self, idx: int, frame: np.ndarray) -> None:
        self.cache[idx] = frame
        self.cache.move_to_end(idx)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def get_frame(self, idx: int) -> np.ndarray | None:
        if idx in self.cache:
            frame = self.cache[idx]
            self.cache.move_to_end(idx)
            return frame

        if idx < 0:
            return None

        if idx == self.last_idx + 1:
            ok, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        self.last_idx = idx
        self._touch_cache(idx, frame)
        return frame

    def close(self) -> None:
        self.cap.release()


def safe_list_get(lst: Any, idx: int, default=None):
    try:
        return lst[idx]
    except Exception:
        return default


def parse_frame_id_from_path(frame_path: str) -> int:
    stem = Path(frame_path).stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(\\d+)$", stem)
    if match:
        return int(match.group(1))
    return -1


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

    return 0


def get_keypoints21(frame_data: dict[str, Any], hand_idx: int) -> np.ndarray | None:
    kps = safe_list_get(frame_data.get("extra_data", []), hand_idx, default=None)
    if kps is None:
        return None
    kps = np.asarray(kps, dtype=float)
    if kps.ndim != 2 or kps.shape[0] < 21 or kps.shape[1] < 2:
        return None
    return kps


def get_depth_at_pixel(depth_img: np.ndarray, u: float, v: float, patch_radius: int = 1):
    h, w = depth_img.shape
    uc = int(round(u))
    vc = int(round(v))
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
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def pixel_to_3d(u: float, v: float, depth_raw: float, intr: dict, depth_scale: float):
    z = float(depth_raw) * float(depth_scale)
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    ppx = float(intr["ppx"])
    ppy = float(intr["ppy"])

    x = (float(u) - ppx) * z / fx
    y = (float(v) - ppy) * z / fy
    return np.array([x, y, z], dtype=float)


def deproject_pixel_like_verify(
    intr: dict,
    u: float,
    v: float,
    depth_m: float,
) -> np.ndarray:
    """Match calibration/2_click_to_base_verify.py pixel->camera 3D logic."""
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    ppx = float(intr["ppx"])
    ppy = float(intr["ppy"])
    x = (float(u) - ppx) * float(depth_m) / fx
    y = (float(v) - ppy) * float(depth_m) / fy
    z = float(depth_m)
    return np.array([x, y, z], dtype=float)


def point3d_to_pixel(point_cam: np.ndarray, intr: dict) -> tuple[float, float] | None:
    x, y, z = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
    if z <= 1e-8:
        return None
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    ppx = float(intr["ppx"])
    ppy = float(intr["ppy"])
    u = fx * x / z + ppx
    v = fy * y / z + ppy
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return float(u), float(v)


def rotmat_to_rpy_zyx(R: np.ndarray) -> tuple[float, float, float]:
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


def draw_hand(frame: np.ndarray, kps: np.ndarray):
    for a, b in HAND_EDGES:
        if a >= len(kps) or b >= len(kps):
            continue
        xa, ya = kps[a, 0], kps[a, 1]
        xb, yb = kps[b, 0], kps[b, 1]
        if np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb):
            cv2.line(
                frame,
                (int(round(xa)), int(round(ya))),
                (int(round(xb)), int(round(yb))),
                (0, 220, 255),
                2,
                cv2.LINE_AA
            )

    for i in range(21):
        x, y = kps[i, 0], kps[i, 1]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        color = (255, 255, 255)
        radius = 3

        if i == KP02:
            color = (0, 255, 0)
            radius = 5
        elif i == KP05:
            color = (255, 0, 255)
            radius = 5

        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)


def put_small_text(img, text, org, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def compute_five_keypoint_plane(
    kps: np.ndarray,
    depth_img: np.ndarray,
    intr: dict,
    depth_scale: float,
    patch_radius: int,
) -> dict[str, Any] | None:
    plane_points_cam = []
    plane_points_uv = []
    for keypoint_index in PLANE_KEYPOINT_INDICES:
        key_u = float(kps[keypoint_index, 0])
        key_v = float(kps[keypoint_index, 1])
        if not (np.isfinite(key_u) and np.isfinite(key_v)):
            return None
        depth_raw = get_depth_at_pixel(depth_img, key_u, key_v, patch_radius=patch_radius)
        if depth_raw is None:
            return None
        depth_m = float(depth_raw) * float(depth_scale)
        point_cam = deproject_pixel_like_verify(intr, key_u, key_v, depth_m)
        plane_points_cam.append(point_cam)
        plane_points_uv.append((key_u, key_v))

    point_array_cam = np.asarray(plane_points_cam, dtype=float)
    center_cam = np.mean(point_array_cam, axis=0)
    centered = point_array_cam - center_cam
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    normal_cam = right_vectors[-1]
    normal_norm = np.linalg.norm(normal_cam)
    if normal_norm < 1e-8:
        return None
    normal_cam = normal_cam / normal_norm
    if normal_cam[2] < 0:
        normal_cam = -normal_cam

    return {
        "points_cam": point_array_cam,
        "points_uv": plane_points_uv,
        "center_cam": center_cam,
        "normal_cam": normal_cam,
    }


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
) -> np.ndarray | None:
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
    centered = inliers - np.mean(inliers, axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    refined_normal = right_vectors[-1]
    refined_norm = np.linalg.norm(refined_normal)
    if refined_norm < 1e-9:
        return None
    return refined_normal / refined_norm


def normal_to_roll_pitch(normal_vec: np.ndarray) -> tuple[float, float]:
    n = np.asarray(normal_vec, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    pitch = math.asin(float(np.clip(nx, -1.0, 1.0)))
    roll = math.atan2(-ny, nz)
    return roll, pitch


def wrap_angle_deg(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def stabilize_angle(
    current_angle: float,
    previous_angle: float | None,
    max_step_deg: float = 12.0,
    ema_alpha: float = 0.35,
) -> float:
    if previous_angle is None:
        return float(current_angle)
    delta = (current_angle - previous_angle + math.pi) % (2.0 * math.pi) - math.pi
    max_step = math.radians(max_step_deg)
    delta = float(np.clip(delta, -max_step, max_step))
    value = previous_angle + delta
    if ema_alpha > 0.0:
        value = (1.0 - ema_alpha) * previous_angle + ema_alpha * value
    # Keep angle bounded to avoid long-term drift outside [-pi, pi].
    value = (value + math.pi) % (2.0 * math.pi) - math.pi
    return float(value)


def draw_plane_overlay(
    frame: np.ndarray,
    plane_info: dict[str, Any],
    intr: dict,
    anchor_uv: tuple[float, float] | None = None,
):
    uv_points = np.asarray(plane_info["points_uv"], dtype=float)
    if uv_points.shape[0] >= 3:
        hull = cv2.convexHull(uv_points.astype(np.float32)).astype(np.int32)
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, hull, (30, 90, 220))
        cv2.addWeighted(overlay, 0.23, frame, 0.77, 0.0, frame)
        cv2.polylines(frame, [hull], isClosed=True, color=(255, 220, 0), thickness=2, lineType=cv2.LINE_AA)

    center_cam = plane_info["center_cam"]
    normal_cam = plane_info["normal_cam"]
    center_uv = point3d_to_pixel(center_cam, intr)
    if anchor_uv is not None:
        center_uv = (float(anchor_uv[0]), float(anchor_uv[1]))
    tip_uv = point3d_to_pixel(center_cam + 0.06 * normal_cam, intr)
    if center_uv is not None and tip_uv is not None:
        c0 = (int(round(center_uv[0])), int(round(center_uv[1])))
        c1 = (int(round(tip_uv[0])), int(round(tip_uv[1])))
        cv2.arrowedLine(frame, c0, c1, (0, 50, 255), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(frame, c0, 4, (0, 50, 255), -1, cv2.LINE_AA)


class Viewer:
    def __init__(
        self,
        frame_reader: VideoFrameReader,
        depth_stack,
        intr,
        depth_scale,
        data,
        T_cam2base,
        hand="right",
        patch_radius=VIS_PATCH_RADIUS,
        save_dir=None,
        export_video_path="annotated_output.mp4",
        export_fps=30.0,
        plane_normal_ema_alpha=0.35,
        plane_normal_max_step_deg=12.0,
    ):
        self.frame_reader = frame_reader
        self.depth_stack = depth_stack
        self.intr = intr
        self.depth_scale = depth_scale
        self.data = data
        self.frame_paths = sorted(data.keys())
        self.frame_id_to_path: dict[int, str] = {}
        for frame_path in self.frame_paths:
            frame_id = parse_frame_id_from_path(frame_path)
            if frame_id >= 0 and frame_id not in self.frame_id_to_path:
                self.frame_id_to_path[frame_id] = frame_path
        self.T_cam2base = T_cam2base
        self.R_cam2base = T_cam2base[:3, :3]
        self.t_cam2base = T_cam2base[:3, 3]
        self.hand = hand
        self.patch_radius = patch_radius
        self.idx = 0
        frame_count = self.frame_reader.frame_count if self.frame_reader.frame_count > 0 else len(self.depth_stack)
        self.length = min(frame_count, len(self.depth_stack), len(self.frame_paths))
        self.window = "Midpoint Trajectory Viewer"
        self.save_dir = Path(save_dir) if save_dir else None
        self.export_video_path = export_video_path
        self.export_fps = export_fps
        self.plane_normal_ema_alpha = float(plane_normal_ema_alpha)
        self.plane_normal_max_step_deg = float(plane_normal_max_step_deg)
        self.previous_plane_normal: np.ndarray | None = None
        self.previous_plane_roll: float | None = None
        self.previous_plane_pitch: float | None = None
        self.clicked_pixel: tuple[int, int] | None = None
        self.measure_first: dict[str, Any] | None = None
        self.measure_last_result: dict[str, Any] | None = None
        self.state_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.state_cache_size = 256

    def _cache_state(self, idx: int, state: dict[str, Any]) -> None:
        self.state_cache[idx] = state
        self.state_cache.move_to_end(idx)
        while len(self.state_cache) > self.state_cache_size:
            self.state_cache.popitem(last=False)

    def set_clicked_pixel(self, x: int, y: int):
        self.clicked_pixel = (int(x), int(y))

    def get_state(self, idx: int):
        if idx in self.state_cache:
            state = self.state_cache[idx]
            self.state_cache.move_to_end(idx)
            return state

        # Prefer exact frame-id alignment: displayed RGB/depth frame idx <-> HaMeR frame id idx.
        # HaMeR frame filenames are typically 1-based (000001.jpg ...),
        # while video/depth arrays are 0-based, so use idx+1 first.
        frame_path = self.frame_id_to_path.get(idx + 1, self.frame_paths[idx])
        frame_data = self.data[frame_path]
        hand_idx = select_hand_index(frame_data, self.hand)

        out = {
            "hand_found": False,
            "kps": None,
            "mid_uv": None,
            "mid_cam": None,
            "mid_base": None,
            "pinch_distance_m": None,
            "gripper_cmd": None,
            "pinch_source": None,
            "base_pitch_deg_global": None,
            "base_roll_deg_global": None,
            "base_pitch_deg_plane": None,
            "base_roll_deg_plane": None,
            "plane_info": None,
        }

        if hand_idx is None:
            self._cache_state(idx, out)
            return out

        kps = get_keypoints21(frame_data, hand_idx)
        if kps is None:
            self._cache_state(idx, out)
            return out

        if not (
            np.isfinite(kps[KP02, 0]) and np.isfinite(kps[KP02, 1]) and
            np.isfinite(kps[KP05, 0]) and np.isfinite(kps[KP05, 1])
        ):
            self._cache_state(idx, out)
            return out

        depth_img = self.depth_stack[idx]

        u2, v2 = float(kps[KP02, 0]), float(kps[KP02, 1])
        u5, v5 = float(kps[KP05, 0]), float(kps[KP05, 1])

        d2 = get_depth_at_pixel(depth_img, u2, v2, self.patch_radius)
        d5 = get_depth_at_pixel(depth_img, u5, v5, self.patch_radius)
        if d2 is None or d5 is None:
            out["kps"] = kps
            self._cache_state(idx, out)
            return out

        p2_cam = deproject_pixel_like_verify(self.intr, u2, v2, float(d2) * float(self.depth_scale))
        p5_cam = deproject_pixel_like_verify(self.intr, u5, v5, float(d5) * float(self.depth_scale))

        # Pinch is defined only by fingertips (kp04-kp08), XY-plane distance only.
        pinch_distance_m = None
        pinch_source = None
        if (
            np.isfinite(kps[KP04, 0]) and np.isfinite(kps[KP04, 1]) and
            np.isfinite(kps[KP08, 0]) and np.isfinite(kps[KP08, 1])
        ):
            u4, v4 = float(kps[KP04, 0]), float(kps[KP04, 1])
            u8, v8 = float(kps[KP08, 0]), float(kps[KP08, 1])
            d4 = get_depth_at_pixel(depth_img, u4, v4, self.patch_radius)
            d8 = get_depth_at_pixel(depth_img, u8, v8, self.patch_radius)
            if d4 is not None and d8 is not None:
                p4_cam = deproject_pixel_like_verify(self.intr, u4, v4, float(d4) * float(self.depth_scale))
                p8_cam = deproject_pixel_like_verify(self.intr, u8, v8, float(d8) * float(self.depth_scale))
                pinch_distance_m = float(np.linalg.norm((p4_cam - p8_cam)[:2]))
                pinch_source = "kp04_kp08"
        gripper_cmd = None
        if pinch_distance_m is not None and DEFAULT_PINCH_OPEN_M > DEFAULT_PINCH_CLOSE_M:
            ratio = (pinch_distance_m - DEFAULT_PINCH_CLOSE_M) / (DEFAULT_PINCH_OPEN_M - DEFAULT_PINCH_CLOSE_M)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            gripper_cmd = DEFAULT_GRIPPER_CMD_MIN + ratio * (DEFAULT_GRIPPER_CMD_MAX - DEFAULT_GRIPPER_CMD_MIN)
        pmid_cam = 0.5 * (p2_cam + p5_cam)
        pmid_base = self.R_cam2base @ pmid_cam + self.t_cam2base

        mid_u = 0.5 * (u2 + u5)
        mid_v = 0.5 * (v2 + v5)

        base_pitch_deg_global = None
        base_roll_deg_global = None
        base_pitch_deg_plane = None
        base_roll_deg_plane = None

        mano_item = safe_list_get(frame_data.get("mano", []), hand_idx, default=None)
        if mano_item is not None and "global_orient" in mano_item:
            try:
                R_cam_hand = np.asarray(mano_item["global_orient"], dtype=float)
                if R_cam_hand.shape == (1, 3, 3):
                    R_cam_hand = R_cam_hand[0]
                R_base_hand = self.R_cam2base @ R_cam_hand
                base_roll, base_pitch, _ = rotmat_to_rpy_zyx(R_base_hand)
                base_pitch_deg_global = float(np.degrees(base_pitch))
                base_roll_deg_global = float(np.degrees(base_roll))
            except Exception:
                pass

        plane_info = compute_five_keypoint_plane(
            kps=kps,
            depth_img=depth_img,
            intr=self.intr,
            depth_scale=self.depth_scale,
            patch_radius=self.patch_radius,
        )
        if plane_info is not None:
            try:
                # Anchor the normal arrow origin at kp02-kp05 midpoint in 3D.
                plane_info["center_cam"] = pmid_cam
                robust_normal = robust_plane_normal_ransac(plane_info["points_cam"])
                if robust_normal is not None:
                    plane_info["normal_cam"] = robust_normal
                stable_normal, self.previous_plane_normal = stabilize_normal(
                    current_normal=plane_info["normal_cam"],
                    previous_normal=self.previous_plane_normal,
                    ema_alpha=self.plane_normal_ema_alpha,
                    max_step_deg=self.plane_normal_max_step_deg,
                )
                plane_info["normal_cam"] = stable_normal
                base_normal = self.R_cam2base @ plane_info["normal_cam"]
                plane_roll, plane_pitch = normal_to_roll_pitch(base_normal)
                plane_roll = stabilize_angle(
                    current_angle=plane_roll,
                    previous_angle=self.previous_plane_roll,
                    max_step_deg=self.plane_normal_max_step_deg,
                    ema_alpha=self.plane_normal_ema_alpha,
                )
                plane_pitch = stabilize_angle(
                    current_angle=plane_pitch,
                    previous_angle=self.previous_plane_pitch,
                    max_step_deg=self.plane_normal_max_step_deg,
                    ema_alpha=self.plane_normal_ema_alpha,
                )
                self.previous_plane_roll = plane_roll
                self.previous_plane_pitch = plane_pitch
                # Keep the displayed plane roll/pitch consistent with extract/replay CSV semantics:
                # base_pitch: up positive / down negative
                # base_roll : left negative / right positive
                internal_roll_offset_deg = wrap_angle_deg(float(np.degrees(plane_roll)) - 180.0)
                internal_pitch_offset_deg = float(np.degrees(plane_pitch))
                base_pitch_deg_plane = -internal_roll_offset_deg
                base_roll_deg_plane = internal_pitch_offset_deg
            except Exception:
                pass

        out["hand_found"] = True
        out["kps"] = kps
        out["mid_uv"] = (mid_u, mid_v)
        out["mid_cam"] = pmid_cam
        out["mid_base"] = pmid_base
        out["pinch_distance_m"] = pinch_distance_m
        out["gripper_cmd"] = gripper_cmd
        out["pinch_source"] = pinch_source
        out["base_pitch_deg_global"] = base_pitch_deg_global
        out["base_roll_deg_global"] = base_roll_deg_global
        out["base_pitch_deg_plane"] = base_pitch_deg_plane
        out["base_roll_deg_plane"] = base_roll_deg_plane
        out["plane_info"] = plane_info
        self._cache_state(idx, out)
        return out

    def draw_frame(self, idx: int):
        frame = self.frame_reader.get_frame(idx)
        if frame is None:
            raise RuntimeError(f"Failed to read RGB frame idx={idx}")
        img = frame.copy()
        state = self.get_state(idx)

        if state["kps"] is not None:
            draw_hand(img, state["kps"])
        if state["plane_info"] is not None:
            draw_plane_overlay(img, state["plane_info"], self.intr, anchor_uv=state["mid_uv"])

        if state["mid_uv"] is not None:
            mu, mv = state["mid_uv"]
            cv2.circle(img, (int(round(mu)), int(round(mv))), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(mu)), int(round(mv))), 11, (255, 255, 255), 2, cv2.LINE_AA)
        if self.clicked_pixel is not None:
            cv2.circle(img, self.clicked_pixel, 6, (0, 0, 255), -1, cv2.LINE_AA)
        if self.measure_first is not None:
            p = self.measure_first["pixel"]
            cv2.circle(img, p, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p, 10, (0, 0, 0), 2, cv2.LINE_AA)

        put_small_text(img, f"frame {idx}", (12, 28), (255, 255, 255))

        if state["mid_base"] is not None:
            bx, by, bz = state["mid_base"]
            put_small_text(img, f"base xyz: {bx:.3f}, {by:.3f}, {bz:.3f}", (12, 56), (0, 255, 0))

        if state["base_pitch_deg_global"] is not None and state["base_roll_deg_global"] is not None:
            put_small_text(
                img,
                f"global pitch: {state['base_pitch_deg_global']:.1f} deg   roll: {state['base_roll_deg_global']:.1f} deg",
                (12, 84),
                (80, 220, 255)
            )
        if state["base_pitch_deg_plane"] is not None and state["base_roll_deg_plane"] is not None:
            put_small_text(
                img,
                f"plane  pitch: {state['base_pitch_deg_plane']:.1f} deg   roll: {state['base_roll_deg_plane']:.1f} deg",
                (12, 112),
                (0, 255, 120)
            )
        if state["pinch_distance_m"] is not None:
            gtxt = "nan" if state["gripper_cmd"] is None else f"{state['gripper_cmd']:.2f}"
            source = state.get("pinch_source", "unknown")
            put_small_text(
                img,
                f"pinch: {state['pinch_distance_m']*1000.0:.1f} mm [{source}]   gripper_cmd: {gtxt}",
                (12, 140),
                (255, 210, 80),
            )

        if state["mid_cam"] is None:
            put_small_text(img, "midpoint invalid", (12, 168), (0, 0, 255))
        elif state["plane_info"] is None:
            put_small_text(img, "plane invalid (occlusion/depth)", (12, 168), (0, 0, 255))
        if self.measure_last_result is not None and self.measure_last_result.get("frame") == idx:
            m = self.measure_last_result
            put_small_text(
                img,
                f"2pt dist: cam3d={m['cam3d_mm']:.1f} mm  base3d={m['base3d_mm']:.1f} mm  xy={m['base_xy_mm']:.1f} mm",
                (12, 196),
                (255, 255, 0),
            )

        return img

    def handle_click_to_base(self, idx: int):
        if self.clicked_pixel is None:
            return
        x, y = self.clicked_pixel
        depth_img = self.depth_stack[idx]
        depth_raw = get_depth_at_pixel(
            depth_img=depth_img,
            u=float(x),
            v=float(y),
            patch_radius=self.patch_radius,
        )
        if depth_raw is None:
            print(f"[WARN] ({x},{y}) 深度无效。")
            return

        # Force the exact same conversion convention as calibration/2_click_to_base_verify.py
        cam_point = deproject_pixel_like_verify(
            self.intr, float(x), float(y), float(depth_raw) * float(self.depth_scale)
        )
        cam_h = np.array([cam_point[0], cam_point[1], cam_point[2], 1.0], dtype=float)
        base_point = self.T_cam2base @ cam_h
        print(
            f"[CLICK] frame={idx} patch={self.patch_radius} "
            f"pixel=({x},{y}) depth={float(depth_raw) * self.depth_scale:.4f}m "
            f"-> base=({base_point[0]:.4f}, {base_point[1]:.4f}, {base_point[2]:.4f})"
        )

        current = {
            "frame": int(idx),
            "pixel": (int(x), int(y)),
            "cam": np.array(cam_point[:3], dtype=float),
            "base": np.array(base_point[:3], dtype=float),
        }
        if self.measure_first is None or self.measure_first["frame"] != int(idx):
            self.measure_first = current
            self.measure_last_result = None
            print(f"[MEASURE] first point set at frame={idx}, pixel=({x},{y})")
            return

        p0 = self.measure_first
        p1 = current
        cam3d = float(np.linalg.norm(p1["cam"] - p0["cam"]))
        base3d = float(np.linalg.norm(p1["base"] - p0["base"]))
        base_xy = float(np.linalg.norm((p1["base"] - p0["base"])[:2]))
        self.measure_last_result = {
            "frame": int(idx),
            "cam3d_mm": cam3d * 1000.0,
            "base3d_mm": base3d * 1000.0,
            "base_xy_mm": base_xy * 1000.0,
        }
        print(
            f"[MEASURE] frame={idx} "
            f"p0={p0['pixel']} p1={p1['pixel']} | "
            f"cam3d={cam3d:.4f}m base3d={base3d:.4f}m base_xy={base_xy:.4f}m"
        )
        # reset for next measurement pair
        self.measure_first = None

    def draw(self):
        return self.draw_frame(self.idx)

    def save_current(self):
        if self.save_dir is None:
            print("[INFO] save_dir is not set, skip saving")
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / f"frame_{self.idx:06d}.png"
        cv2.imwrite(str(out_path), self.draw())
        print(f"[INFO] saved {out_path}")

    def export_video(self, output_path: str, fps: float = 30.0, start_idx: int = 0):
        if self.length <= 0:
            print("[WARN] no frames to export")
            return

        start_idx = max(0, min(start_idx, self.length - 1))
        first = self.draw_frame(start_idx)
        h, w = first.shape[:2]

        out_path = Path(output_path)
        if out_path.parent and str(out_path.parent) != ".":
            out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            print(f"[ERROR] failed to open video writer: {out_path}")
            return

        print(f"[INFO] exporting video to: {out_path}")
        print(f"[INFO] export range: {start_idx} -> {self.length - 1}")

        for i in range(start_idx, self.length):
            frame = self.draw_frame(i)
            writer.write(frame)

            if ((i - start_idx + 1) % 30 == 0) or (i == self.length - 1):
                print(f"[INFO] export progress: {i - start_idx + 1}/{self.length - start_idx}")

        writer.release()
        print(f"[INFO] export done: {out_path}")

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        h, w = self.frame_reader.frame_shape[:2]
        fx = float(self.intr["fx"])
        fy = float(self.intr["fy"])
        ppx = float(self.intr["ppx"])
        ppy = float(self.intr["ppy"])
        print(
            "[INFO] Viewer calibration: "
            f"frame_size={w}x{h}, "
            f"fx={fx:.3f}, fy={fy:.3f}, ppx={ppx:.3f}, ppy={ppy:.3f}, "
            f"depth_scale={float(self.depth_scale):.6f} m/unit"
        )
        print(
            "[INFO] Quick scale hints: "
            f"fx/width={fx/max(1,w):.4f}, fy/height={fy/max(1,h):.4f}"
        )

        def on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.set_clicked_pixel(x, y)
                self.handle_click_to_base(self.idx)

        cv2.setMouseCallback(self.window, on_mouse)

        while True:
            canvas = self.draw()
            cv2.imshow(self.window, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord("a"):
                self.idx = max(0, self.idx - 1)
            elif key == ord("d"):
                self.idx = min(self.length - 1, self.idx + 1)
            elif key == ord("s"):
                self.save_current()
            elif key == ord("v"):
                self.export_video(self.export_video_path, self.export_fps, start_idx=0)
            elif key == ord("V"):
                self.export_video(self.export_video_path, self.export_fps, start_idx=self.idx)

        cv2.destroyAllWindows()
        self.frame_reader.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--episode-dir", required=True)
    p.add_argument(
        "--transform-json",
        default=str(get_default_camera_to_base_path()),
        help="contains T_cam2base; default uses calibration/outputs/cam2base_latest.json",
    )
    p.add_argument("--rgb-path", default=None, help="optional path to rgb video, e.g. 4.16.mp4")
    p.add_argument("--hand", choices=["right", "left", "first"], default="right")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--export-video-path", default="data/visualizations/annotated_output.mp4")
    p.add_argument("--export-fps", type=float, default=DEFAULT_VISUALIZE_EXPORT_FPS)
    p.add_argument("--export-only", action="store_true", help="export mp4 directly without opening GUI")
    return p.parse_args()


def resolve_rgb_path(episode_dir: Path, rgb_path_argument: str | None) -> Path:
    if rgb_path_argument is not None:
        return Path(rgb_path_argument)

    direct_rgb_path = episode_dir / "rgb.mp4"
    if direct_rgb_path.exists():
        return direct_rgb_path

    # Support new layout:
    # episode_dir = data/raw/depth/<cam_key>/chunk-xxx/file-xxx
    # rgb      = data/raw/videos/observation.images.<cam_key>/chunk-xxx/file-xxx.mp4
    parts = episode_dir.parts
    if "depth" in parts:
        depth_index = parts.index("depth")
        if depth_index + 3 < len(parts):
            camera_key = parts[depth_index + 1]
            chunk_name = parts[depth_index + 2]
            file_stem = parts[depth_index + 3]
            dataset_root = Path(*parts[:depth_index]) if depth_index > 0 else Path("/")
            rgb_candidate = (
                dataset_root
                / "videos"
                / f"observation.images.{camera_key}"
                / chunk_name
                / f"{file_stem}.mp4"
            )
            if rgb_candidate.exists():
                return rgb_candidate

    raise FileNotFoundError(
        "RGB video not found. Pass --rgb-path explicitly, or use episode-dir layout "
        "that contains rgb.mp4 (legacy) / maps from depth/<cam>/chunk/file to "
        "videos/observation.images.<cam>/chunk/file.mp4 (new)."
    )


def main():
    args = parse_args()

    episode_dir = Path(args.episode_dir)

    rgb_path = resolve_rgb_path(episode_dir=episode_dir, rgb_path_argument=args.rgb_path)
    depth_path = episode_dir / "depth.npz"
    intr_path = episode_dir / "intrinsics.json"

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB video not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    if not intr_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intr_path}")

    frame_reader = VideoFrameReader(str(rgb_path), cache_size=128)
    depth_stack = load_depth_npz(str(depth_path))
    intr, depth_scale = load_intrinsics_json(str(intr_path))
    data = load_pkl(args.input_pkl)
    transform_path = str(Path(args.transform_json).expanduser().resolve())
    T_cam2base = load_transform_json(transform_path)
    print(f"[INFO] using transform-json: {transform_path}")

    viewer = Viewer(
        frame_reader=frame_reader,
        depth_stack=depth_stack,
        intr=intr,
        depth_scale=depth_scale,
        data=data,
        T_cam2base=T_cam2base,
        hand=args.hand,
        patch_radius=VIS_PATCH_RADIUS,
        save_dir=args.save_dir,
        export_video_path=args.export_video_path,
        export_fps=args.export_fps,
        plane_normal_ema_alpha=0.35,
        plane_normal_max_step_deg=12.0,
    )
    try:
        if args.export_only:
            viewer.export_video(args.export_video_path, args.export_fps, start_idx=0)
            return
        viewer.run()
    finally:
        viewer.frame_reader.close()


if __name__ == "__main__":
    main()
