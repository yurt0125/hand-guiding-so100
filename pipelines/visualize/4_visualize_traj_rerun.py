#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
import time
import termios
import tty
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import rerun as rr
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Rerun is required. Install with: pip install rerun-sdk"
    ) from exc

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from common.config import (  # noqa: E402
    DEFAULT_VISUALIZE_EXPORT_FPS,
    get_default_camera_to_base_path,
)

KP02 = 2
KP04 = 4
KP05 = 5
KP08 = 8
PLANE_KEYPOINT_INDICES = (2, 3, 4, 5, 6, 7, 8)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class YoloSegHelper:
    def __init__(self, model_path: str, conf: float, target_class: str, device: str):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is required for --enable-yolo-seg") from exc
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.target_class = str(target_class)
        self.device = str(device)

    def infer(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
        confs = r.boxes.conf.detach().cpu().numpy()
        xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(int)
        names = r.names

        masks_np = None
        if r.masks is not None and r.masks.data is not None:
            masks_np = r.masks.data.detach().cpu().numpy()  # [N,H,W]

        for i, cid in enumerate(cls_ids):
            cname = names.get(cid, str(cid))
            if cname != self.target_class:
                continue
            det: dict[str, Any] = {
                "class_name": cname,
                "conf": float(confs[i]),
                "bbox": tuple(map(int, xyxy[i].tolist())),
                "mask": None,
            }
            if masks_np is not None and i < masks_np.shape[0]:
                det["mask"] = masks_np[i] > 0.5
            out.append(det)
        return out


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


def rpy_zyx_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    return rz @ ry @ rx


def to_float(row: dict[str, str], key: str) -> float | None:
    if key not in row:
        return None
    value = row[key]
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def load_extract_csv_rows(csv_path: Path) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    by_frame_index: dict[int, dict[str, str]] = {}
    by_frame_id: dict[int, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx = to_float(row, "frame_index")
            if idx is not None:
                by_frame_index[int(idx)] = row
            fid = to_float(row, "frame_id")
            if fid is not None:
                by_frame_id[int(fid)] = row
    return by_frame_index, by_frame_id


def resolve_rgb_path(episode_dir: Path, rgb_path_argument: str | None) -> Path:
    if rgb_path_argument is not None:
        return Path(rgb_path_argument)
    direct_rgb_path = episode_dir / "rgb.mp4"
    if direct_rgb_path.exists():
        return direct_rgb_path
    parts = episode_dir.parts
    if "depth" in parts:
        depth_index = parts.index("depth")
        if depth_index + 3 < len(parts):
            camera_key = parts[depth_index + 1]
            chunk_name = parts[depth_index + 2]
            file_stem = parts[depth_index + 3]
            dataset_root = Path(*parts[:depth_index]) if depth_index > 0 else Path("/")
            rgb_candidate = dataset_root / "videos" / f"observation.images.{camera_key}" / chunk_name / f"{file_stem}.mp4"
            if rgb_candidate.exists():
                return rgb_candidate
    raise FileNotFoundError("RGB video not found. Pass --rgb-path explicitly.")


def build_state(
    frame_data: dict[str, Any],
    csv_row: dict[str, str] | None,
    intr: dict,
    hand: str = "right",
):
    out = {
        "hand_found": False,
        "kps": None,
        "mid_uv": None,
        "mid_cam": None,
        "mid_base": None,
        "pinch_distance_m": None,
        "gripper_cmd": None,
        "pinch_source": None,
        "base_pitch_deg_plane": None,
        "base_roll_deg_plane": None,
        "plane_points_uv": None,
        "plane_normal_cam": None,
        "plane_center_uv": None,
        "plane_tip_uv": None,
        "episode_index": None,
    }
    hand_idx = select_hand_index(frame_data, hand)
    if hand_idx is None:
        return out
    kps = get_keypoints21(frame_data, hand_idx)
    if kps is None:
        return out
    if not (
        np.isfinite(kps[KP02, 0]) and np.isfinite(kps[KP02, 1]) and
        np.isfinite(kps[KP05, 0]) and np.isfinite(kps[KP05, 1])
    ):
        return out
    u2, v2 = float(kps[KP02, 0]), float(kps[KP02, 1])
    u5, v5 = float(kps[KP05, 0]), float(kps[KP05, 1])

    plane_points_uv = []
    for kp_idx in PLANE_KEYPOINT_INDICES:
        ku, kv = float(kps[kp_idx, 0]), float(kps[kp_idx, 1])
        if not (np.isfinite(ku) and np.isfinite(kv)):
            continue
        plane_points_uv.append((ku, kv))

    mid_cam = None
    mid_base = None
    pinch_distance_m = None
    gripper_cmd = None
    pinch_source = None
    base_pitch_deg_plane = None
    base_roll_deg_plane = None
    plane_normal_cam = None
    if csv_row is not None:
        epi = to_float(csv_row, "episode_index")
        if epi is not None:
            out["episode_index"] = int(epi)
        mx = to_float(csv_row, "mid_cam_x")
        my = to_float(csv_row, "mid_cam_y")
        mz = to_float(csv_row, "mid_cam_z")
        if mx is not None and my is not None and mz is not None:
            mid_cam = np.asarray([mx, my, mz], dtype=float)
        bx = to_float(csv_row, "mid_base_x")
        by = to_float(csv_row, "mid_base_y")
        bz = to_float(csv_row, "mid_base_z")
        if bx is not None and by is not None and bz is not None:
            mid_base = np.asarray([bx, by, bz], dtype=float)
        pinch_distance_m = to_float(csv_row, "pinch_distance_m")
        gripper_cmd = to_float(csv_row, "gripper_cmd")
        pinch_source_val = csv_row.get("pinch_source", "")
        pinch_source = pinch_source_val if pinch_source_val else None
        base_pitch_deg_plane = to_float(csv_row, "base_pitch")
        base_roll_deg_plane = to_float(csv_row, "base_roll")
        cam_roll = to_float(csv_row, "cam_roll")
        cam_pitch = to_float(csv_row, "cam_pitch")
        cam_yaw = to_float(csv_row, "cam_yaw")
        if cam_roll is not None and cam_pitch is not None and cam_yaw is not None:
            rotation_cam = rpy_zyx_to_matrix(cam_roll, cam_pitch, cam_yaw)
            plane_normal_cam = rotation_cam[:, 2]

    out.update(
        {
            "hand_found": True,
            "kps": kps,
            "mid_uv": (0.5 * (u2 + u5), 0.5 * (v2 + v5)),
            "mid_cam": mid_cam,
            "mid_base": mid_base,
            "pinch_distance_m": pinch_distance_m,
            "gripper_cmd": gripper_cmd,
            "pinch_source": pinch_source,
            "base_pitch_deg_plane": base_pitch_deg_plane,
            "base_roll_deg_plane": base_roll_deg_plane,
            "plane_points_uv": plane_points_uv,
            "plane_normal_cam": plane_normal_cam,
            "plane_center_uv": None,
            "plane_tip_uv": None,
        }
    )
    if plane_normal_cam is not None and out["mid_cam"] is not None:
        center_uv = point3d_to_pixel(np.asarray(out["mid_cam"], dtype=float), intr)
        tip_uv = point3d_to_pixel(
            np.asarray(out["mid_cam"], dtype=float) + 0.06 * np.asarray(plane_normal_cam, dtype=float),
            intr,
        )
        out["plane_center_uv"] = center_uv
        out["plane_tip_uv"] = tip_uv
    return out


def log_frame_to_rerun(
    step_idx: int,
    frame_idx: int,
    frame_bgr: np.ndarray | None,
    state: dict[str, Any],
):
    # Use monotonically increasing step timeline so each key press always produces a new visible frame.
    rr.set_time("step", sequence=step_idx)
    rr.set_time("frame", sequence=frame_idx)

    if frame_bgr is not None:
        rr.log("cam/image", rr.Image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    if state.get("pinch_distance_m") is not None:
        rr.log("signals/pinch_distance_m", rr.Scalars(state["pinch_distance_m"]))
    if state.get("gripper_cmd") is not None:
        rr.log("signals/gripper_cmd", rr.Scalars(state["gripper_cmd"]))
    if state.get("base_pitch_deg_plane") is not None:
        rr.log("signals/base_pitch_deg_plane", rr.Scalars(state["base_pitch_deg_plane"]))
    if state.get("base_roll_deg_plane") is not None:
        rr.log("signals/base_roll_deg_plane", rr.Scalars(state["base_roll_deg_plane"]))
    rr.log("signals/frame_index", rr.Scalars(float(frame_idx)))
    rr.log("status/info", rr.TextDocument(build_status_text(frame_idx, state), media_type=rr.MediaType.MARKDOWN))


def put_small_text(img: np.ndarray, text: str, org: tuple[int, int], color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def build_status_text(frame_idx: int, state: dict[str, Any]) -> str:
    lines = [f"# Frame {frame_idx}", ""]
    if state.get("episode_index") is not None:
        lines.append(f"### Episode Index: **{int(state['episode_index'])}**")
    valid = (
        state.get("hand_found", False)
        and (
            state.get("mid_base") is not None
            or (
                state.get("base_pitch_deg_plane") is not None
                and state.get("base_roll_deg_plane") is not None
            )
            or state.get("pinch_distance_m") is not None
        )
    )
    if not valid:
        lines.append("## Invalid")
        return "\n".join(lines)

    if state.get("mid_base") is not None:
        bx, by, bz = state["mid_base"]
        lines.append(f"### Base X (m): **{bx:.3f}**")
        lines.append(f"### Base Y (m): **{by:.3f}**")
        lines.append(f"### Base Z (m): **{bz:.3f}**")
    if state.get("base_pitch_deg_plane") is not None and state.get("base_roll_deg_plane") is not None:
        lines.append(f"### Plane Pitch (deg): **{state['base_pitch_deg_plane']:.1f}**")
        lines.append(f"### Plane Roll (deg): **{state['base_roll_deg_plane']:.1f}**")
    if state.get("pinch_distance_m") is not None:
        gtxt = "nan" if state.get("gripper_cmd") is None else f"{state['gripper_cmd']:.2f}"
        lines.append(f"### Pinch Distance (mm): **{state['pinch_distance_m']*1000.0:.1f}**")
        lines.append(f"### Gripper Cmd: **{gtxt}**")
    if state.get("yolo_target_count") is not None:
        lines.append(f"### YOLO Target Count: **{int(state['yolo_target_count'])}**")
    return "\n".join(lines)


def draw_state_overlay(
    frame: np.ndarray,
    state: dict[str, Any],
    frame_idx: int,
    yolo_dets: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    img = frame.copy()
    if yolo_dets:
        for det in yolo_dets:
            mask = det.get("mask", None)
            if mask is not None:
                color = np.zeros_like(img)
                color[:, :] = (0, 255, 0)
                img[mask] = cv2.addWeighted(img, 0.45, color, 0.55, 0)[mask]
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{det['class_name']} {det['conf']:.2f}"
            put_small_text(img, label, (x1, max(20, y1 - 8)), color=(0, 255, 255), scale=0.55)

    if state.get("plane_points_uv") and len(state["plane_points_uv"]) >= 3:
        uv_points = np.asarray(state["plane_points_uv"], dtype=float)
        hull = cv2.convexHull(uv_points.astype(np.float32)).astype(np.int32)
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, hull, (30, 90, 220))
        cv2.addWeighted(overlay, 0.23, img, 0.77, 0.0, img)
        cv2.polylines(img, [hull], isClosed=True, color=(255, 220, 0), thickness=2, lineType=cv2.LINE_AA)
    center_uv = state.get("plane_center_uv", None)
    tip_uv = state.get("plane_tip_uv", None)
    if center_uv is not None and tip_uv is not None:
        c0 = (int(round(center_uv[0])), int(round(center_uv[1])))
        c1 = (int(round(tip_uv[0])), int(round(tip_uv[1])))
        cv2.arrowedLine(img, c0, c1, (0, 50, 255), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(img, c0, 4, (0, 50, 255), -1, cv2.LINE_AA)

    if state["kps"] is not None:
        kps = state["kps"]
        for a, b in HAND_EDGES:
            if a < len(kps) and b < len(kps):
                xa, ya = kps[a, 0], kps[a, 1]
                xb, yb = kps[b, 0], kps[b, 1]
                if np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb):
                    cv2.line(
                        img,
                        (int(round(xa)), int(round(ya))),
                        (int(round(xb)), int(round(yb))),
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
        for i in range(min(21, len(kps))):
            x, y = kps[i, 0], kps[i, 1]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 200, 0), -1, cv2.LINE_AA)

    if state["mid_uv"] is not None:
        mu, mv = state["mid_uv"]
        cv2.circle(img, (int(round(mu)), int(round(mv))), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(mu)), int(round(mv))), 11, (255, 255, 255), 2, cv2.LINE_AA)

    # Text metrics are logged to a separate Rerun panel (status/info).
    return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--csv-path", required=True, help="CSV generated by pipelines/extract/3_extract_traj.py")
    p.add_argument("--episode-dir", required=True)
    p.add_argument(
        "--transform-json",
        default=str(get_default_camera_to_base_path()),
        help="contains T_cam2base; default uses calibration/outputs/cam2base_latest.json",
    )
    p.add_argument("--rgb-path", default=None)
    p.add_argument("--hand", choices=["right", "left", "first"], default="right")
    p.add_argument("--fps", type=float, default=DEFAULT_VISUALIZE_EXPORT_FPS)
    p.add_argument("--max-frames", type=int, default=-1, help="debug limit; -1 means all")
    p.add_argument("--start-frame", type=int, default=0, help="Start logging from this frame index.")
    p.add_argument(
        "--realtime-fps",
        type=float,
        default=-1.0,
        help="Playback FPS in auto mode. <0 means follow --fps; 0 means as fast as possible.",
    )
    p.add_argument("--step-through", action="store_true", help="Manual stepping: Enter next frame, q quit.")
    p.add_argument("--save-rrd", default=None, help="Optional output .rrd path")
    p.add_argument("--enable-yolo-seg", action="store_true", help="Run YOLO-seg on each RGB frame.")
    p.add_argument("--yolo-model", type=str, default="yolov8n-seg.pt")
    p.add_argument("--yolo-target-class", type=str, default="banana")
    p.add_argument("--yolo-conf", type=float, default=0.35)
    p.add_argument("--yolo-device", type=str, default="0")
    return p.parse_args()


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def read_single_key() -> str:
    """Read one key from terminal without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def start_global_key_listener() -> tuple[deque[str], Any]:
    queue: deque[str] = deque(maxlen=64)
    try:
        from pynput import keyboard  # type: ignore
    except Exception:
        return queue, None

    def on_press(key):
        try:
            ch = key.char
            if ch is not None:
                queue.append(ch)
        except Exception:
            if key == keyboard.Key.esc:
                queue.append("\x1b")

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return queue, listener


def poll_global_key(global_q: deque[str]) -> str | None:
    if global_q:
        return global_q.popleft()
    return None


def main():
    args = parse_args()
    yolo_seg: YoloSegHelper | None = None
    if args.enable_yolo_seg:
        yolo_seg = YoloSegHelper(
            model_path=args.yolo_model,
            conf=args.yolo_conf,
            target_class=args.yolo_target_class,
            device=args.yolo_device,
        )
        print(
            f"[INFO] YOLO-seg enabled: model={args.yolo_model}, "
            f"class={args.yolo_target_class}, conf={args.yolo_conf}, device={args.yolo_device}"
        )

    episode_dir = Path(args.episode_dir)
    rgb_path = resolve_rgb_path(episode_dir=episode_dir, rgb_path_argument=args.rgb_path)
    intr_path = episode_dir / "intrinsics.json"
    if not intr_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intr_path}")

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    by_frame_index, by_frame_id = load_extract_csv_rows(csv_path)
    if not by_frame_index and not by_frame_id:
        raise RuntimeError(f"No valid frame rows found in CSV: {csv_path}")

    intr, depth_scale = load_intrinsics_json(str(intr_path))
    data = load_pkl(args.input_pkl)
    T_cam2base = load_transform_json(str(Path(args.transform_json).expanduser().resolve()))

    app_id = f"traj_visualize_rerun_{uuid.uuid4().hex[:8]}"
    rr.init(app_id, spawn=True, recording_id=str(uuid.uuid4()))
    if args.save_rrd:
        rr.save(args.save_rrd)

    cap = None
    total_video_frames = 10**12
    if rgb_path.exists():
        cap = cv2.VideoCapture(str(rgb_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {rgb_path}")
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        print(f"[WARN] RGB video not found, continue without image overlay: {rgb_path}")
    frame_paths = sorted(data.keys())
    frame_id_to_path = {}
    for frame_path in frame_paths:
        frame_id = parse_frame_id_from_path(frame_path)
        if frame_id >= 0 and frame_id not in frame_id_to_path:
            frame_id_to_path[frame_id] = frame_path

    end_frame_exclusive = min(total_video_frames, len(frame_paths))
    start_frame = max(0, int(args.start_frame))
    if start_frame >= end_frame_exclusive:
        raise ValueError(
            f"--start-frame={start_frame} is out of range. Valid frame index: [0, {end_frame_exclusive - 1}]"
        )
    if args.max_frames > 0:
        end_frame_exclusive = min(end_frame_exclusive, start_frame + int(args.max_frames))

    total = end_frame_exclusive - start_frame
    print(f"[INFO] Logging {total} frames to Rerun from start-frame={start_frame}...")

    step_counter = 0
    global_key_q, global_key_listener = start_global_key_listener()
    try:
        if args.step_through:
            mode = "terminal+global" if global_key_listener is not None else "terminal"
            print(f"[INFO] Step mode controls ({mode}): A=prev, D=next, Q/ESC/Ctrl+C=quit")
            idx = start_frame
            last_logged_idx = None
            while True:
                if last_logged_idx != idx:
                    frame = None
                    if cap is not None:
                        frame = read_frame_at(cap, idx)
                        if frame is None:
                            print(f"[WARN] failed to decode frame={idx}")
                            break
                    frame_path = frame_id_to_path.get(idx + 1, frame_paths[idx])
                    frame_data = data[frame_path]
                    csv_row = by_frame_id.get(idx + 1, by_frame_index.get(idx))
                    state = build_state(
                        frame_data=frame_data,
                        csv_row=csv_row,
                        intr=intr,
                        hand=args.hand,
                    )
                    yolo_dets = yolo_seg.infer(frame) if (yolo_seg is not None and frame is not None) else None
                    if yolo_dets is not None:
                        state["yolo_target_count"] = len(yolo_dets)
                    annotated = (
                        draw_state_overlay(frame, state, idx, yolo_dets=yolo_dets)
                        if frame is not None
                        else None
                    )
                    log_frame_to_rerun(step_counter, idx, annotated, state)
                    step_counter += 1
                    last_logged_idx = idx
                key = poll_global_key(global_key_q)
                if key is None:
                    key = read_single_key()
                if key in ("q", "Q", "\x1b", "\x03"):
                    break
                if key in ("a", "A"):
                    idx = max(start_frame, idx - 1)
                    continue
                if key in ("d", "D"):
                    idx = min(end_frame_exclusive - 1, idx + 1)
                    continue
        else:
            effective_realtime_fps = args.fps if args.realtime_fps < 0 else args.realtime_fps
            frame_interval_s = (1.0 / effective_realtime_fps) if effective_realtime_fps > 0 else 0.0
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
            for idx in range(start_frame, end_frame_exclusive):
                frame = None
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                frame_path = frame_id_to_path.get(idx + 1, frame_paths[idx])
                frame_data = data[frame_path]
                csv_row = by_frame_id.get(idx + 1, by_frame_index.get(idx))
                state = build_state(
                    frame_data=frame_data,
                    csv_row=csv_row,
                    intr=intr,
                    hand=args.hand,
                )
                yolo_dets = yolo_seg.infer(frame) if (yolo_seg is not None and frame is not None) else None
                if yolo_dets is not None:
                    state["yolo_target_count"] = len(yolo_dets)
                annotated = (
                    draw_state_overlay(frame, state, idx, yolo_dets=yolo_dets)
                    if frame is not None
                    else None
                )
                log_frame_to_rerun(step_counter, idx, annotated, state)
                step_counter += 1
                if frame_interval_s > 0:
                    time.sleep(frame_interval_s)
                progressed = idx - start_frame + 1
                if (progressed % 120 == 0) or (idx == end_frame_exclusive - 1):
                    epi = state.get("episode_index")
                    epi_txt = f", episode={int(epi)}" if epi is not None else ""
                    print(f"[INFO] Progress: {progressed}/{total}{epi_txt}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C, exiting.")

    if cap is not None:
        cap.release()
    if global_key_listener is not None:
        try:
            global_key_listener.stop()
        except Exception:
            pass
    print("[INFO] Done.")
    if args.save_rrd:
        print(f"[INFO] Saved RRD: {args.save_rrd}")


if __name__ == "__main__":
    main()
