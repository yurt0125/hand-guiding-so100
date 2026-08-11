#!/usr/bin/env python3
"""
Record episodic RGB-D datasets from an Intel RealSense D435.

Design goals:
- Similar operator flow to LeRobot's record loop: record episodes, reset, re-record, stop.
- Save episode-oriented datasets with synchronized RGB, depth, timestamps, and camera intrinsics.
- Keep dependencies small: pyrealsense2, numpy, opencv-python.

Controls
--------
SPACE : start/stop current episode
x     : discard current buffered episode
q/ESC : quit
h     : toggle help overlay
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import sys
import threading

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import (
    DEFAULT_RECORD_DEPTH_MAX_M,
    DEFAULT_RECORD_DEPTH_MIN_M,
    DEFAULT_RECORD_EPISODE_TIME_S,
    DEFAULT_RECORD_FPS,
    DEFAULT_RECORD_HEIGHT,
    DEFAULT_RECORD_NUM_EPISODES,
    DEFAULT_RECORD_RESET_TIME_S,
    DEFAULT_RECORD_WARMUP_FRAMES,
    DEFAULT_RECORD_WIDTH,
)

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise SystemExit(
        "pyrealsense2 is required. Install librealsense and pyrealsense2 first."
    ) from e


@dataclass
class RecorderConfig:
    root: str
    fps: int = DEFAULT_RECORD_FPS
    width: int = DEFAULT_RECORD_WIDTH
    height: int = DEFAULT_RECORD_HEIGHT
    episode_time_s: float = DEFAULT_RECORD_EPISODE_TIME_S
    reset_time_s: float = DEFAULT_RECORD_RESET_TIME_S
    num_episodes: int = DEFAULT_RECORD_NUM_EPISODES
    task: str = "D435 episode recording"
    warmup_frames: int = DEFAULT_RECORD_WARMUP_FRAMES
    depth_max_m: float = DEFAULT_RECORD_DEPTH_MAX_M
    depth_min_m: float = DEFAULT_RECORD_DEPTH_MIN_M
    serial: str | None = None
    save_preview: bool = True
    top_camera_source: str | None = None
    top_camera_width: int = DEFAULT_RECORD_WIDTH
    top_camera_height: int = DEFAULT_RECORD_HEIGHT
    top_camera_fps: int = DEFAULT_RECORD_FPS


class D435EpisodeRecorder:
    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg
        self.root = Path(cfg.root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        self.profile = None
        self.depth_scale = None
        self.device_name = None
        self.serial = None
        self.intrinsics = None
        self.top_capture: cv2.VideoCapture | None = None
        self.top_thread: threading.Thread | None = None
        self.top_running = False
        self.top_lock = threading.Lock()
        self.top_latest_frame: np.ndarray | None = None
        self.top_latest_timestamp_s: float | None = None

        self.recording = False
        self.running = True
        self.show_help = True

        self.episode_index = self._next_episode_index()
        self.saved_episodes = 0

        self.rgb_frames: list[np.ndarray] = []
        self.depth_frames_mm: list[np.ndarray] = []
        self.timestamps_s: list[float] = []
        self.frame_meta: list[dict[str, Any]] = []

        self.top_rgb_frames: list[np.ndarray] = []
        self.top_timestamps_s: list[float] = []
        self.top_frame_meta: list[dict[str, Any]] = []

        self.record_started_t = None
        self.reset_started_t = None
        self.in_reset = False

    def _next_episode_index(self) -> int:
        existing = sorted(self.episodes_dir.glob("episode_*"))
        if not existing:
            return 0
        nums = []
        for p in existing:
            try:
                nums.append(int(p.name.split("_")[-1]))
            except Exception:
                pass
        return (max(nums) + 1) if nums else 0

    def connect(self):
        config = rs.config()
        if self.cfg.serial:
            config.enable_device(self.cfg.serial)
        config.enable_stream(rs.stream.color, self.cfg.width, self.cfg.height, rs.format.bgr8, self.cfg.fps)
        config.enable_stream(rs.stream.depth, self.cfg.width, self.cfg.height, rs.format.z16, self.cfg.fps)

        self.profile = self.pipeline.start(config)

        device = self.profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.serial = device.get_info(rs.camera_info.serial_number)

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.intrinsics = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }

        for _ in range(self.cfg.warmup_frames):
            self.pipeline.wait_for_frames()

        if self.cfg.top_camera_source:
            self.top_capture = cv2.VideoCapture(self.cfg.top_camera_source)
            if not self.top_capture.isOpened():
                raise RuntimeError(f"Failed to open top camera source: {self.cfg.top_camera_source}")
            self.top_capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.top_camera_width))
            self.top_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.top_camera_height))
            self.top_capture.set(cv2.CAP_PROP_FPS, float(self.cfg.top_camera_fps))
            self.top_running = True
            self.top_thread = threading.Thread(target=self._top_capture_loop, daemon=True)
            self.top_thread.start()

        self._write_root_metadata()

    def disconnect(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.top_running = False
        if self.top_thread is not None:
            self.top_thread.join(timeout=1.0)
        if self.top_capture is not None:
            self.top_capture.release()

    def _top_capture_loop(self):
        while self.top_running and self.top_capture is not None:
            ok, frame = self.top_capture.read()
            if not ok:
                time.sleep(0.002)
                continue
            timestamp_s = time.time()
            with self.top_lock:
                self.top_latest_frame = frame
                self.top_latest_timestamp_s = timestamp_s

    def _write_root_metadata(self):
        meta = {
            "dataset_type": "d435_episode_dataset",
            "task": self.cfg.task,
            "fps": self.cfg.fps,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "device_name": self.device_name,
            "serial": self.serial,
            "depth_scale_m_per_unit": self.depth_scale,
            "intrinsics": self.intrinsics,
            "top_camera": {
                "enabled": bool(self.cfg.top_camera_source),
                "source": self.cfg.top_camera_source,
                "width": self.cfg.top_camera_width,
                "height": self.cfg.top_camera_height,
                "fps": self.cfg.top_camera_fps,
            },
            "created_unix_s": time.time(),
            "controls": {
                "space": "start/stop episode",
                "x": "discard current episode buffer",
                "q_or_esc": "quit",
                "h": "toggle help overlay",
            },
        }
        with open(self.root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def clear_buffers(self):
        self.rgb_frames.clear()
        self.depth_frames_mm.clear()
        self.timestamps_s.clear()
        self.frame_meta.clear()

        self.top_rgb_frames.clear()
        self.top_timestamps_s.clear()
        self.top_frame_meta.clear()

    def start_episode(self):
        self.clear_buffers()
        self.recording = True
        self.in_reset = False
        self.record_started_t = time.perf_counter()
        print(f"[INFO] Started episode {self.episode_index:06d}")

    def stop_episode(self, save: bool = True):
        self.recording = False
        self.record_started_t = None
        if save and len(self.rgb_frames) > 0:
            self._save_episode()
            self.saved_episodes += 1
            self.episode_index += 1
            self.in_reset = True
            self.reset_started_t = time.perf_counter()
        else:
            print("[INFO] Episode discarded or empty.")
        self.clear_buffers()

    def _save_episode(self):
        ep_dir = self.episodes_dir / f"episode_{self.episode_index:06d}"
        ep_dir.mkdir(parents=True, exist_ok=False)

        d435_dir = ep_dir / "d435"
        top_dir = ep_dir / "top"
        calib_dir = ep_dir / "calib"
        d435_dir.mkdir(parents=True, exist_ok=True)
        top_dir.mkdir(parents=True, exist_ok=True)
        calib_dir.mkdir(parents=True, exist_ok=True)

        h, w = self.rgb_frames[0].shape[:2]
        rgb_path = d435_dir / "rgb.mp4"
        writer = cv2.VideoWriter(str(rgb_path), cv2.VideoWriter_fourcc(*"mp4v"), self.cfg.fps, (w, h))
        for frame in self.rgb_frames:
            writer.write(frame)
        writer.release()

        depth_stack = np.stack(self.depth_frames_mm, axis=0).astype(np.uint16)
        np.savez_compressed(d435_dir / "depth.npz", depth_mm=depth_stack)
        np.save(d435_dir / "timestamps.npy", np.asarray(self.timestamps_s, dtype=np.float64))

        with open(d435_dir / "frame_meta.jsonl", "w", encoding="utf-8") as f:
            for row in self.frame_meta:
                f.write(json.dumps(row) + "\n")

        with open(calib_dir / "intrinsics_d435.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "intrinsics": self.intrinsics,
                    "depth_scale_m_per_unit": self.depth_scale,
                },
                f,
                indent=2,
            )

        if len(self.top_rgb_frames) > 0:
            top_h, top_w = self.top_rgb_frames[0].shape[:2]
            top_rgb_path = top_dir / "rgb.mp4"
            top_writer = cv2.VideoWriter(
                str(top_rgb_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.cfg.top_camera_fps,
                (top_w, top_h),
            )
            for frame in self.top_rgb_frames:
                top_writer.write(frame)
            top_writer.release()
            np.save(top_dir / "timestamps.npy", np.asarray(self.top_timestamps_s, dtype=np.float64))
            with open(top_dir / "frame_meta.jsonl", "w", encoding="utf-8") as f:
                for row in self.top_frame_meta:
                    f.write(json.dumps(row) + "\n")

            with open(calib_dir / "intrinsics_top.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "width": self.cfg.top_camera_width,
                        "height": self.cfg.top_camera_height,
                        "fps": self.cfg.top_camera_fps,
                        "source": self.cfg.top_camera_source,
                    },
                    f,
                    indent=2,
                )

        if self.cfg.save_preview:
            preview = self._make_contact_sheet()
            if preview is not None:
                cv2.imwrite(str(ep_dir / "preview.jpg"), preview)

        episode_meta = {
            "episode_index": self.episode_index,
            "num_frames": len(self.rgb_frames),
            "fps": self.cfg.fps,
            "duration_s": len(self.rgb_frames) / float(self.cfg.fps),
            "task": self.cfg.task,
            "saved_unix_s": time.time(),
            "has_top_camera": len(self.top_rgb_frames) > 0,
            "top_num_frames": len(self.top_rgb_frames),
            "top_fps": self.cfg.top_camera_fps if len(self.top_rgb_frames) > 0 else None,
        }
        with open(ep_dir / "episode_meta.json", "w", encoding="utf-8") as f:
            json.dump(episode_meta, f, indent=2)

        print(f"[INFO] Saved episode {self.episode_index:06d} with {len(self.rgb_frames)} frames to {ep_dir}")

    def _make_contact_sheet(self):
        if len(self.rgb_frames) == 0:
            return None
        idxs = np.linspace(0, len(self.rgb_frames) - 1, num=min(6, len(self.rgb_frames)), dtype=int)
        thumbs = []
        for i in idxs:
            img = self.rgb_frames[i]
            thumb = cv2.resize(img, (320, 180))
            cv2.putText(
                thumb,
                f"frame {i}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)
        if len(thumbs) <= 3:
            return np.hstack(thumbs)
        row1 = np.hstack(thumbs[:3])
        row2 = np.hstack(thumbs[3:])
        if row2.shape[1] < row1.shape[1]:
            pad = np.zeros((row2.shape[0], row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
            row2 = np.hstack([row2, pad])
        return np.vstack([row1, row2])

    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            return None, None, None

        depth = self.spatial.process(depth)
        depth = self.temporal.process(depth)
        depth = self.hole_filling.process(depth)

        color_img = np.asanyarray(color.get_data())
        depth_raw = np.asanyarray(depth.get_data())

        t = time.time()
        meta = {
            "color_frame_number": int(color.get_frame_number()),
            "depth_frame_number": int(depth.get_frame_number()),
            "color_timestamp_ms": float(color.get_timestamp()),
            "depth_timestamp_ms": float(depth.get_timestamp()),
        }
        return color_img, depth_raw, (t, meta)

    def get_top_frame(self):
        if self.top_capture is None:
            return None, None
        with self.top_lock:
            if self.top_latest_frame is None or self.top_latest_timestamp_s is None:
                return None, None
            frame = self.top_latest_frame.copy()
            timestamp_s = float(self.top_latest_timestamp_s)
        if frame is None:
            return None, None
        return frame, timestamp_s

    def depth_to_preview(self, depth_raw: np.ndarray) -> np.ndarray:
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        depth_m = np.clip(depth_m, self.cfg.depth_min_m, self.cfg.depth_max_m)
        depth_norm = (depth_m - self.cfg.depth_min_m) / max(1e-6, self.cfg.depth_max_m - self.cfg.depth_min_m)
        depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    def maybe_append_frame(self, color_img, depth_raw, timestamp_s, meta, top_frame, top_timestamp_s):
        if not self.recording:
            return
        self.rgb_frames.append(color_img.copy())
        self.depth_frames_mm.append(depth_raw.copy())
        self.timestamps_s.append(float(timestamp_s))
        self.frame_meta.append(
            {
                "frame_index": len(self.rgb_frames) - 1,
                "timestamp_s": float(timestamp_s),
                **meta,
            }
        )

        if top_frame is not None and top_timestamp_s is not None:
            self.top_rgb_frames.append(top_frame.copy())
            self.top_timestamps_s.append(float(top_timestamp_s))
            self.top_frame_meta.append(
                {
                    "frame_index": len(self.top_rgb_frames) - 1,
                    "timestamp_s": float(top_timestamp_s),
                }
            )

    def update_auto_transitions(self):
        if self.recording and self.record_started_t is not None:
            elapsed = time.perf_counter() - self.record_started_t
            if elapsed >= self.cfg.episode_time_s:
                self.stop_episode(save=True)

        if self.in_reset and self.reset_started_t is not None:
            if (time.perf_counter() - self.reset_started_t) >= self.cfg.reset_time_s:
                self.in_reset = False
                self.reset_started_t = None

        if self.saved_episodes >= self.cfg.num_episodes:
            self.running = False

    def draw_overlay(self, color_img: np.ndarray, depth_preview: np.ndarray, top_frame: np.ndarray | None) -> np.ndarray:
        if top_frame is None:
            canvas = np.hstack([color_img, depth_preview])
        else:
            top_resized = cv2.resize(top_frame, (color_img.shape[1], color_img.shape[0]))
            canvas = np.hstack([color_img, depth_preview, top_resized])

        lines = [
            f"Task: {self.cfg.task}",
            f"Device: {self.device_name} ({self.serial})",
            f"Episode index: {self.episode_index:06d}",
            f"Saved episodes: {self.saved_episodes}/{self.cfg.num_episodes}",
            f"Recording: {'YES' if self.recording else 'NO'}",
            f"Buffered frames: {len(self.rgb_frames)}",
        ]

        if self.recording and self.record_started_t is not None:
            lines.append(
                f"Episode elapsed: {time.perf_counter() - self.record_started_t:5.1f}s / {self.cfg.episode_time_s:.1f}s"
            )
        elif self.in_reset and self.reset_started_t is not None:
            remaining = max(0.0, self.cfg.reset_time_s - (time.perf_counter() - self.reset_started_t))
            lines.append(f"Reset window: {remaining:4.1f}s")

        lines.append("Press h to toggle controls")
        if self.show_help:
            lines += [
                "SPACE=start/stop episode",
                "x=discard current episode",
                "q or ESC=quit",
            ]

        y = 28
        for line in lines:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y += 28

        label = "RGB | DEPTH" if top_frame is None else "RGB | DEPTH | TOP_RGB"
        cv2.putText(canvas, label, (12, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def run(self):
        self.connect()
        print("[INFO] Recorder ready.")
        print("[INFO] Controls: SPACE=start/stop, x=discard, q=quit, h=help")

        try:
            while self.running:
                color_img, depth_raw, payload = self.get_aligned_frames()
                if color_img is None:
                    continue

                timestamp_s, meta = payload
                top_frame, top_timestamp_s = self.get_top_frame()

                self.maybe_append_frame(color_img, depth_raw, timestamp_s, meta, top_frame, top_timestamp_s)
                self.update_auto_transitions()

                depth_preview = self.depth_to_preview(depth_raw)
                vis = self.draw_overlay(color_img, depth_preview, top_frame)
                cv2.imshow("D435 Episode Recorder", vis)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    if self.recording:
                        print("[INFO] Quitting. Current buffered episode will be discarded.")
                    self.running = False
                elif key == ord(" "):
                    if self.recording:
                        self.stop_episode(save=True)
                    elif not self.in_reset and self.saved_episodes < self.cfg.num_episodes:
                        self.start_episode()
                elif key == ord("x"):
                    if self.recording:
                        self.stop_episode(save=False)
                    else:
                        self.clear_buffers()
                        print("[INFO] Cleared current buffers.")
                elif key == ord("h"):
                    self.show_help = not self.show_help
        finally:
            cv2.destroyAllWindows()
            self.disconnect()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--fps", type=int, default=DEFAULT_RECORD_FPS)
    p.add_argument("--width", type=int, default=DEFAULT_RECORD_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_RECORD_HEIGHT)
    p.add_argument("--episode-time-s", type=float, default=DEFAULT_RECORD_EPISODE_TIME_S)
    p.add_argument("--reset-time-s", type=float, default=DEFAULT_RECORD_RESET_TIME_S)
    p.add_argument("--num-episodes", type=int, default=DEFAULT_RECORD_NUM_EPISODES)
    p.add_argument("--task", type=str, default="D435 episode recording")
    p.add_argument("--warmup-frames", type=int, default=DEFAULT_RECORD_WARMUP_FRAMES)
    p.add_argument("--depth-max-m", type=float, default=DEFAULT_RECORD_DEPTH_MAX_M)
    p.add_argument("--depth-min-m", type=float, default=DEFAULT_RECORD_DEPTH_MIN_M)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--top-camera-source", type=str, default=None)
    p.add_argument("--top-camera-width", type=int, default=DEFAULT_RECORD_WIDTH)
    p.add_argument("--top-camera-height", type=int, default=DEFAULT_RECORD_HEIGHT)
    p.add_argument("--top-camera-fps", type=int, default=DEFAULT_RECORD_FPS)
    args = p.parse_args()
    return RecorderConfig(
        root=args.root,
        fps=args.fps,
        width=args.width,
        height=args.height,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        num_episodes=args.num_episodes,
        task=args.task,
        warmup_frames=args.warmup_frames,
        depth_max_m=args.depth_max_m,
        depth_min_m=args.depth_min_m,
        serial=args.serial,
        save_preview=not args.no_preview,
        top_camera_source=args.top_camera_source,
        top_camera_width=args.top_camera_width,
        top_camera_height=args.top_camera_height,
        top_camera_fps=args.top_camera_fps,
    )


def main():
    cfg = parse_args()
    print(json.dumps(asdict(cfg), indent=2))
    recorder = D435EpisodeRecorder(cfg)
    recorder.run()


if __name__ == "__main__":
    main()
