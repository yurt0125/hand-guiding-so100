#!/usr/bin/env python3
"""
Camera-only episode recorder based on official LeRobot camera classes.

- D435 via RealSenseCamera (RGB + depth)
- Optional top RGB via OpenCVCamera
- No robot control / no teleop required

Output layout (per episode):
  episode_xxxxxx/
    d435/rgb.mp4
    d435/depth.npz
    d435/timestamps.npy
    d435/frame_meta.jsonl
    top/rgb.mp4                     (if enabled)
    top/timestamps.npy              (if enabled)
    top/frame_meta.jsonl            (if enabled)
    calib/intrinsics_d435.json
    calib/intrinsics_top.json       (if enabled)
    episode_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = PROJECT_ROOT / "third-party" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.cameras.configs import ColorMode  # type: ignore
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera  # type: ignore
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # type: ignore
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera  # type: ignore
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # type: ignore


@dataclass
class RecordConfig:
    root: str
    fps: int
    width: int
    height: int
    episode_time_s: float
    reset_time_s: float
    num_episodes: int
    warmup_frames: int
    d435_serial_or_name: str
    top_camera_source: str | None
    top_camera_width: int
    top_camera_height: int
    top_camera_fps: int
    task: str


class LeRobotCameraOnlyRecorder:
    def __init__(self, config: RecordConfig):
        self.config = config
        self.root = Path(config.root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self.d435_camera: RealSenseCamera | None = None
        self.top_camera: OpenCVCamera | None = None

        self.running = True
        self.recording = False
        self.in_reset = False

        self.episode_index = self._next_episode_index()
        self.saved_episodes = 0
        self.record_started_time: float | None = None
        self.reset_started_time: float | None = None

        self.rgb_frames: list[np.ndarray] = []
        self.depth_frames: list[np.ndarray] = []
        self.timestamps_s: list[float] = []
        self.frame_meta: list[dict[str, Any]] = []

        self.top_frames: list[np.ndarray] = []
        self.top_timestamps_s: list[float] = []
        self.top_frame_meta: list[dict[str, Any]] = []

    def _next_episode_index(self) -> int:
        existing = sorted(self.episodes_dir.glob("episode_*"))
        if not existing:
            return 0
        numeric_indices = []
        for path in existing:
            try:
                numeric_indices.append(int(path.name.split("_")[-1]))
            except Exception:
                continue
        return (max(numeric_indices) + 1) if numeric_indices else 0

    def connect(self) -> None:
        d435_config = RealSenseCameraConfig(
            serial_number_or_name=self.config.d435_serial_or_name,
            fps=self.config.fps,
            width=self.config.width,
            height=self.config.height,
            color_mode=ColorMode.BGR,
            use_depth=True,
            warmup_s=max(0.1, self.config.warmup_frames / max(1.0, float(self.config.fps))),
        )
        self.d435_camera = RealSenseCamera(d435_config)
        self.d435_camera.connect()

        if self.config.top_camera_source:
            top_index_or_path: int | str
            source_value = self.config.top_camera_source
            if source_value.startswith("/dev/video"):
                top_index_or_path = source_value
            else:
                try:
                    top_index_or_path = int(source_value)
                except ValueError:
                    top_index_or_path = source_value

            top_config = OpenCVCameraConfig(
                index_or_path=top_index_or_path,
                fps=self.config.top_camera_fps,
                width=self.config.top_camera_width,
                height=self.config.top_camera_height,
                color_mode=ColorMode.BGR,
                warmup_s=0.2,
            )
            self.top_camera = OpenCVCamera(top_config)
            self.top_camera.connect()

        self._write_root_metadata()

    def disconnect(self) -> None:
        if self.d435_camera is not None:
            self.d435_camera.disconnect()
        if self.top_camera is not None:
            self.top_camera.disconnect()

    def _write_root_metadata(self) -> None:
        assert self.d435_camera is not None
        d435_intrinsics = {
            "width": self.d435_camera.width,
            "height": self.d435_camera.height,
            "fps": self.d435_camera.fps,
            "serial_number": self.d435_camera.serial_number,
        }
        metadata = {
            "dataset_type": "camera_only_lerobot",
            "task": self.config.task,
            "created_unix_s": time.time(),
            "d435": d435_intrinsics,
            "top_camera": {
                "enabled": self.top_camera is not None,
                "source": self.config.top_camera_source,
                "width": self.config.top_camera_width,
                "height": self.config.top_camera_height,
                "fps": self.config.top_camera_fps,
            },
        }
        with open(self.root / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def clear_buffers(self) -> None:
        self.rgb_frames.clear()
        self.depth_frames.clear()
        self.timestamps_s.clear()
        self.frame_meta.clear()
        self.top_frames.clear()
        self.top_timestamps_s.clear()
        self.top_frame_meta.clear()

    def start_episode(self) -> None:
        self.clear_buffers()
        self.recording = True
        self.in_reset = False
        self.record_started_time = time.perf_counter()
        print(f"[INFO] Started episode {self.episode_index:06d}")

    def stop_episode(self, save: bool) -> None:
        self.recording = False
        self.record_started_time = None

        if save and self.rgb_frames:
            self._save_episode()
            self.saved_episodes += 1
            self.episode_index += 1
            self.in_reset = True
            self.reset_started_time = time.perf_counter()
        else:
            print("[INFO] Episode discarded or empty.")

        self.clear_buffers()

    def _save_episode(self) -> None:
        episode_dir = self.episodes_dir / f"episode_{self.episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=False)

        d435_dir = episode_dir / "d435"
        top_dir = episode_dir / "top"
        calib_dir = episode_dir / "calib"
        d435_dir.mkdir(parents=True, exist_ok=True)
        calib_dir.mkdir(parents=True, exist_ok=True)
        if self.top_frames:
            top_dir.mkdir(parents=True, exist_ok=True)

        height, width = self.rgb_frames[0].shape[:2]
        d435_writer = cv2.VideoWriter(
            str(d435_dir / "rgb.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.config.fps,
            (width, height),
        )
        for frame in self.rgb_frames:
            d435_writer.write(frame)
        d435_writer.release()

        depth_array = np.stack(self.depth_frames, axis=0).astype(np.uint16)
        np.savez_compressed(d435_dir / "depth.npz", depth_mm=depth_array)
        np.save(d435_dir / "timestamps.npy", np.asarray(self.timestamps_s, dtype=np.float64))

        with open(d435_dir / "frame_meta.jsonl", "w", encoding="utf-8") as file:
            for row in self.frame_meta:
                file.write(json.dumps(row) + "\n")

        d435_intrinsics = {
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "serial_or_name": self.config.d435_serial_or_name,
        }
        with open(calib_dir / "intrinsics_d435.json", "w", encoding="utf-8") as file:
            json.dump(d435_intrinsics, file, indent=2)

        if self.top_frames:
            top_height, top_width = self.top_frames[0].shape[:2]
            top_writer = cv2.VideoWriter(
                str(top_dir / "rgb.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.config.top_camera_fps,
                (top_width, top_height),
            )
            for frame in self.top_frames:
                top_writer.write(frame)
            top_writer.release()

            np.save(top_dir / "timestamps.npy", np.asarray(self.top_timestamps_s, dtype=np.float64))
            with open(top_dir / "frame_meta.jsonl", "w", encoding="utf-8") as file:
                for row in self.top_frame_meta:
                    file.write(json.dumps(row) + "\n")

            top_intrinsics = {
                "source": self.config.top_camera_source,
                "width": self.config.top_camera_width,
                "height": self.config.top_camera_height,
                "fps": self.config.top_camera_fps,
            }
            with open(calib_dir / "intrinsics_top.json", "w", encoding="utf-8") as file:
                json.dump(top_intrinsics, file, indent=2)

        episode_metadata = {
            "episode_index": self.episode_index,
            "num_frames": len(self.rgb_frames),
            "fps": self.config.fps,
            "duration_s": len(self.rgb_frames) / float(max(1, self.config.fps)),
            "saved_unix_s": time.time(),
            "has_top_camera": bool(self.top_frames),
            "top_num_frames": len(self.top_frames),
        }
        with open(episode_dir / "episode_meta.json", "w", encoding="utf-8") as file:
            json.dump(episode_metadata, file, indent=2)

        print(f"[INFO] Saved episode {self.episode_index:06d} -> {episode_dir}")

    def _update_auto_transitions(self) -> None:
        if self.recording and self.record_started_time is not None:
            elapsed = time.perf_counter() - self.record_started_time
            if elapsed >= self.config.episode_time_s:
                self.stop_episode(save=True)

        if self.in_reset and self.reset_started_time is not None:
            if (time.perf_counter() - self.reset_started_time) >= self.config.reset_time_s:
                self.in_reset = False
                self.reset_started_time = None

        if self.saved_episodes >= self.config.num_episodes:
            self.running = False

    def _draw_overlay(
        self,
        d435_color: np.ndarray,
        d435_depth_mm: np.ndarray,
        top_color: np.ndarray | None,
    ) -> np.ndarray:
        depth_preview = cv2.convertScaleAbs(d435_depth_mm, alpha=0.03)
        depth_preview = cv2.applyColorMap(depth_preview, cv2.COLORMAP_JET)

        if top_color is None:
            canvas = np.hstack([d435_color, depth_preview])
            label = "D435_RGB | D435_DEPTH"
        else:
            top_resized = cv2.resize(top_color, (d435_color.shape[1], d435_color.shape[0]))
            canvas = np.hstack([d435_color, depth_preview, top_resized])
            label = "D435_RGB | D435_DEPTH | TOP_RGB"

        lines = [
            f"Episode: {self.episode_index:06d}",
            f"Saved: {self.saved_episodes}/{self.config.num_episodes}",
            f"Recording: {'YES' if self.recording else 'NO'}",
            f"Buffered frames: {len(self.rgb_frames)}",
            "SPACE=start/stop  x=discard  q/ESC=quit",
        ]

        y = 28
        for line in lines:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y += 28

        cv2.putText(
            canvas,
            label,
            (12, canvas.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def run(self) -> None:
        self.connect()
        assert self.d435_camera is not None

        print("[INFO] LeRobot camera-only recorder ready.")
        print("[INFO] Controls: SPACE=start/stop, x=discard, q/ESC=quit")

        try:
            while self.running:
                try:
                    d435_color = self.d435_camera.read(timeout_ms=200)
                    d435_depth = self.d435_camera.read_depth(timeout_ms=200)
                except Exception:
                    continue

                top_color: np.ndarray | None = None
                if self.top_camera is not None:
                    try:
                        top_color = self.top_camera.async_read(timeout_ms=100)
                    except Exception:
                        top_color = None

                frame_timestamp_s = time.time()
                if self.recording:
                    frame_index = len(self.rgb_frames)
                    self.rgb_frames.append(d435_color.copy())
                    self.depth_frames.append(d435_depth.copy())
                    self.timestamps_s.append(frame_timestamp_s)
                    self.frame_meta.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_s": frame_timestamp_s,
                        }
                    )

                    if top_color is not None:
                        top_frame_index = len(self.top_frames)
                        self.top_frames.append(top_color.copy())
                        self.top_timestamps_s.append(frame_timestamp_s)
                        self.top_frame_meta.append(
                            {
                                "frame_index": top_frame_index,
                                "timestamp_s": frame_timestamp_s,
                            }
                        )

                self._update_auto_transitions()

                visualization = self._draw_overlay(d435_color, d435_depth, top_color)
                cv2.imshow("LeRobot Camera-only Recorder", visualization)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    self.running = False
                elif key == ord(" "):
                    if self.recording:
                        self.stop_episode(save=True)
                    elif not self.in_reset and self.saved_episodes < self.config.num_episodes:
                        self.start_episode()
                elif key == ord("x"):
                    if self.recording:
                        self.stop_episode(save=False)
                    else:
                        self.clear_buffers()

        finally:
            cv2.destroyAllWindows()
            self.disconnect()


def parse_arguments() -> RecordConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--episode-time-s", type=float, default=15.0)
    parser.add_argument("--reset-time-s", type=float, default=3.0)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--task", type=str, default="hand demo collection")
    parser.add_argument("--d435-serial-or-name", type=str, required=True)
    parser.add_argument("--top-camera-source", type=str, default=None)
    parser.add_argument("--top-camera-width", type=int, default=640)
    parser.add_argument("--top-camera-height", type=int, default=480)
    parser.add_argument("--top-camera-fps", type=int, default=30)

    args = parser.parse_args()
    return RecordConfig(
        root=args.root,
        fps=args.fps,
        width=args.width,
        height=args.height,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        num_episodes=args.num_episodes,
        warmup_frames=args.warmup_frames,
        task=args.task,
        d435_serial_or_name=args.d435_serial_or_name,
        top_camera_source=args.top_camera_source,
        top_camera_width=args.top_camera_width,
        top_camera_height=args.top_camera_height,
        top_camera_fps=args.top_camera_fps,
    )


def main() -> None:
    config = parse_arguments()
    recorder = LeRobotCameraOnlyRecorder(config)
    recorder.run()


if __name__ == "__main__":
    main()
