#!/usr/bin/env python3
"""
Local wrapper around official LeRobot record script.

Goal:
- Keep official lerobot_record pipeline unchanged as much as possible.
- Add depth observation for so101_follower without modifying third-party sources.

Usage example:
PYTHONPATH=third-party/lerobot/src:. python pipelines/record/3_rawdata_record.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=left \
  --robot.cameras="{'d435': {'type': 'intelrealsense', 'serial_number_or_name': '216322074780', 'width': 640, 'height': 480, 'fps': 15, 'use_depth': true}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader \
  --dataset.repo_id=rita/test_depth \
  --dataset.root=/abs/path/to/new_dataset_dir \
  --dataset.video=false \
  --dataset.push_to_hub=false \
  --resume=false
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.utils import make_robot_from_config as official_make_robot_from_config
from lerobot.scripts import lerobot_record as official_record
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import register_third_party_plugins

CAMERA_ONLY_MODE = False
CAPTURED_DEPTH_CAMERA_METADATA: dict[str, dict[str, Any]] = {}
CAPTURED_DEPTH_FRAMES: dict[str, list[np.ndarray]] = {}
LATEST_DEPTH_PREVIEW: dict[str, np.ndarray] = {}
DEPTH_CAPTURE_ENABLED = True
DEPTH_EPISODE_SNAPSHOTS: list[dict[str, int]] = []
DEPTH_EPISODE_WRITE_INDEX: dict[str, int] = {}


def _set_depth_capture_enabled(enabled: bool) -> None:
    global DEPTH_CAPTURE_ENABLED
    DEPTH_CAPTURE_ENABLED = bool(enabled)


def _snapshot_depth_lengths() -> dict[str, int]:
    return {key: len(value) for key, value in CAPTURED_DEPTH_FRAMES.items()}


def _restore_depth_lengths(snapshot: dict[str, int]) -> None:
    for key, frames in list(CAPTURED_DEPTH_FRAMES.items()):
        keep = int(snapshot.get(key, 0))
        if keep < len(frames):
            CAPTURED_DEPTH_FRAMES[key] = frames[:keep]


def _ensure_camera_depth_dirs(dataset_root: Path, camera_key: str) -> tuple[Path, Path]:
    camera_root = dataset_root / "depth" / camera_key
    episodes_root = camera_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    return camera_root, episodes_root


def _next_episode_index(dataset_root: Path, camera_key: str) -> int:
    if camera_key in DEPTH_EPISODE_WRITE_INDEX:
        return DEPTH_EPISODE_WRITE_INDEX[camera_key]
    _, episodes_root = _ensure_camera_depth_dirs(dataset_root, camera_key)
    existing = sorted(episodes_root.glob("episode-*.npz"))
    if not existing:
        DEPTH_EPISODE_WRITE_INDEX[camera_key] = 0
        return 0
    last = existing[-1].stem  # episode-000123
    idx = int(last.split("-")[-1]) + 1
    DEPTH_EPISODE_WRITE_INDEX[camera_key] = idx
    return idx


def _write_episode_depth_from_snapshot(dataset_root: Path, snapshot: dict[str, int]) -> None:
    """
    Write per-episode depth directly from in-memory depth buffer, no chunk/parquet dependency.
    """
    for camera_key, depth_frames in CAPTURED_DEPTH_FRAMES.items():
        start = int(snapshot.get(camera_key, 0))
        end = len(depth_frames)
        if end <= start:
            continue
        segment = np.stack(depth_frames[start:end], axis=0).astype(np.uint16)
        camera_root, episodes_root = _ensure_camera_depth_dirs(dataset_root, camera_key)
        episode_idx = _next_episode_index(dataset_root, camera_key)
        np.savez_compressed(episodes_root / f"episode-{episode_idx:06d}.npz", depth_mm=segment)
        DEPTH_EPISODE_WRITE_INDEX[camera_key] = episode_idx + 1

        # Keep one canonical intrinsics file per camera.
        metadata = CAPTURED_DEPTH_CAMERA_METADATA.get(camera_key)
        if metadata is not None:
            (camera_root / "intrinsics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"[INFO] wrote episode depth: {episodes_root / f'episode-{episode_idx:06d}.npz'} "
            f"frames={segment.shape[0]}"
        )


def _build_merged_depth_from_episodes(dataset_root: Path) -> None:
    depth_root = dataset_root / "depth"
    if not depth_root.exists():
        return
    for camera_dir in sorted(depth_root.glob("*")):
        if not camera_dir.is_dir():
            continue
        episodes_root = camera_dir / "episodes"
        if not episodes_root.exists():
            continue
        episode_paths = sorted(episodes_root.glob("episode-*.npz"))
        if not episode_paths:
            continue

        all_parts: list[np.ndarray] = []
        index_rows: list[dict[str, int]] = []
        cursor = 0
        for ep_path in episode_paths:
            ep_idx = int(ep_path.stem.split("-")[-1])
            ep_depth = np.load(ep_path)["depth_mm"]
            start = cursor
            end = cursor + int(ep_depth.shape[0])
            all_parts.append(ep_depth)
            index_rows.append({"episode_index": ep_idx, "start": start, "end": end})
            cursor = end

        merged = np.concatenate(all_parts, axis=0).astype(np.uint16)
        np.savez_compressed(camera_dir / "depth.npz", depth_mm=merged)
        (camera_dir / "depth_index.json").write_text(
            json.dumps(
                {
                    "camera_key": camera_dir.name,
                    "total_frames": int(merged.shape[0]),
                    "episodes": index_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (episodes_root / "manifest.json").write_text(
            json.dumps(
                {
                    "camera_key": camera_dir.name,
                    "episodes": [
                        {
                            "episode_index": int(p.stem.split("-")[-1]),
                            "path": str(p.relative_to(dataset_root)),
                            "frames": int(np.load(p)["depth_mm"].shape[0]),
                        }
                        for p in episode_paths
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[INFO] built merged depth: {camera_dir / 'depth.npz'} total_frames={merged.shape[0]}")


def _ensure_depth_processors(camera: Any) -> dict[str, Any]:
    import pyrealsense2 as rs  # local import

    processors = getattr(camera, "_depth_processors", None)
    if processors is None:
        processors = {
            "align": rs.align(rs.stream.color),
            "spatial": rs.spatial_filter(),
            "temporal": rs.temporal_filter(),
            "hole": rs.hole_filling_filter(),
        }
        setattr(camera, "_depth_processors", processors)
    return processors


def _read_aligned_rgb_depth(camera: Any, camera_key: str, retries: int = 3, timeout_ms: int = 400):
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            ret, frameset = camera.rs_pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
            if not ret or frameset is None:
                continue
            processors = _ensure_depth_processors(camera)
            aligned = processors["align"].process(frameset)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if depth_frame is None or color_frame is None:
                continue
            depth_frame = processors["spatial"].process(depth_frame)
            depth_frame = processors["temporal"].process(depth_frame)
            depth_frame = processors["hole"].process(depth_frame)
            return color_frame, depth_frame
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"{camera_key}: failed to read aligned RGB+Depth frames ({last_error})") from last_error
    raise RuntimeError(f"{camera_key}: failed to read aligned RGB+Depth frames.")


def _depth_to_preview_bgr(depth_mm: np.ndarray) -> np.ndarray:
    """Convert uint16 depth(mm) to a BGR colormap image for rerun preview only."""
    depth_u16 = np.asarray(depth_mm, dtype=np.uint16)
    valid = depth_u16 > 0
    if not np.any(valid):
        return np.zeros((depth_u16.shape[0], depth_u16.shape[1], 3), dtype=np.uint8)
    valid_values = depth_u16[valid]
    near = float(np.percentile(valid_values, 2))
    far = float(np.percentile(valid_values, 98))
    if far <= near:
        far = near + 1.0
    clipped = np.clip(depth_u16.astype(np.float32), near, far)
    scaled = ((clipped - near) * (255.0 / (far - near))).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _extract_cli_value(arguments: list[str], key: str) -> str | None:
    inline_prefix = f"{key}="
    for index, argument in enumerate(arguments):
        if argument.startswith(inline_prefix):
            return argument[len(inline_prefix) :]
        if argument == key and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _has_cli_key(arguments: list[str], key: str) -> bool:
    inline_prefix = f"{key}="
    for argument in arguments:
        if argument == key or argument.startswith(inline_prefix):
            return True
    return False


def _has_cli_prefix(arguments: list[str], key_prefix: str) -> bool:
    for argument in arguments:
        if argument.startswith(f"{key_prefix}=") or argument == key_prefix:
            return True
    return False


def _parse_cameras_config(cameras_text: str) -> dict[str, Any]:
    # 1) Try strict JSON first.
    try:
        parsed_json = json.loads(cameras_text)
        if isinstance(parsed_json, dict):
            return parsed_json
    except Exception:
        pass

    # 2) Try Python literal with tolerant boolean/null normalization.
    normalized_text = re.sub(r"\btrue\b", "True", cameras_text)
    normalized_text = re.sub(r"\bfalse\b", "False", normalized_text)
    normalized_text = re.sub(r"\bnull\b", "None", normalized_text)
    try:
        parsed_literal = ast.literal_eval(normalized_text)
        if not isinstance(parsed_literal, dict):
            raise ValueError("parsed value is not a dict")
        return parsed_literal
    except Exception as exception:
        raise ValueError(f"Failed to parse --robot.cameras: {exception}") from exception


class CameraOnlyRobotWithDepth(Robot):
    """Camera-only robot adapter that keeps official lerobot_record pipeline."""

    name = "hand"
    config_class = SO101Follower.config_class

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for camera_key, camera_config in self.config.cameras.items():
            features[camera_key] = (camera_config.height, camera_config.width, 3)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def connect(self, calibrate: bool = True) -> None:
        for camera in self.cameras.values():
            camera.connect()
        self._capture_depth_camera_metadata()
        self._is_connected = True

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            camera.disconnect()
        self._is_connected = False

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        observation_dict: dict[str, Any] = {}
        for camera_key, camera in self.cameras.items():
            if getattr(camera, "use_depth", False) and hasattr(camera, "rs_pipeline"):
                color_frame, depth_frame = _read_aligned_rgb_depth(camera, camera_key)

                color_raw = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                color_image = camera._postprocess_image(color_raw, depth_frame=False)
                observation_dict[camera_key] = color_image
                LATEST_DEPTH_PREVIEW[camera_key] = _depth_to_preview_bgr(depth_image)
                if DEPTH_CAPTURE_ENABLED:
                    CAPTURED_DEPTH_FRAMES.setdefault(camera_key, []).append(depth_image.copy())
            else:
                observation_dict[camera_key] = camera.async_read()

        return observation_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _capture_depth_camera_metadata(self) -> None:
        for camera_key, camera in self.cameras.items():
            if not getattr(camera, "use_depth", False):
                continue
            metadata = extract_depth_camera_metadata(camera)
            if metadata is not None:
                CAPTURED_DEPTH_CAMERA_METADATA[camera_key] = metadata


class SO101FollowerWithDepth(SO101Follower):
    """SO101 follower that exposes depth frames as additional observations."""

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        camera_features: dict[str, tuple] = {}
        for camera_key in self.cameras:
            camera_config = self.config.cameras[camera_key]
            camera_features[camera_key] = (camera_config.height, camera_config.width, 3)
        return camera_features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()
        observation_dict = self.bus.sync_read("Present_Position")
        observation_dict = {f"{motor}.pos": value for motor, value in observation_dict.items()}
        _ = (time.perf_counter() - start_time) * 1e3

        for camera_key, camera in self.cameras.items():
            if getattr(camera, "use_depth", False) and hasattr(camera, "rs_pipeline"):
                color_frame, depth_frame = _read_aligned_rgb_depth(camera, camera_key)

                color_raw = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())  # uint16 depth in mm

                color_image = camera._postprocess_image(color_raw, depth_frame=False)
                observation_dict[camera_key] = color_image
                LATEST_DEPTH_PREVIEW[camera_key] = _depth_to_preview_bgr(depth_image)
                if DEPTH_CAPTURE_ENABLED:
                    CAPTURED_DEPTH_FRAMES.setdefault(camera_key, []).append(depth_image.copy())
            else:
                # Keep default behavior for non-depth cameras.
                observation_dict[camera_key] = camera.async_read()

        return observation_dict

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        for camera_key, camera in self.cameras.items():
            if not getattr(camera, "use_depth", False):
                continue
            metadata = extract_depth_camera_metadata(camera)
            if metadata is not None:
                CAPTURED_DEPTH_CAMERA_METADATA[camera_key] = metadata


def extract_depth_camera_metadata(camera: Any) -> dict[str, Any] | None:
    """Extract color/depth intrinsics + depth scale from a connected RealSense camera."""
    try:
        if not hasattr(camera, "rs_profile") or camera.rs_profile is None:
            return None
        if not hasattr(camera, "rs_pipeline") or camera.rs_pipeline is None:
            return None
        import pyrealsense2 as rs  # local import to avoid hard dependency at module import time

        depth_stream_profile = camera.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream_profile = camera.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intrinsics = depth_stream_profile.get_intrinsics()
        color_intrinsics = color_stream_profile.get_intrinsics()
        depth_sensor = camera.rs_profile.get_device().first_depth_sensor()
        depth_scale_m_per_unit = float(depth_sensor.get_depth_scale())
        return {
            # Keep this field as the default projection intrinsics for downstream tools.
            # Pixel coordinates come from RGB images, so COLOR intrinsics must be used.
            "intrinsics": {
                "fx": float(color_intrinsics.fx),
                "fy": float(color_intrinsics.fy),
                "ppx": float(color_intrinsics.ppx),
                "ppy": float(color_intrinsics.ppy),
                "width": int(color_intrinsics.width),
                "height": int(color_intrinsics.height),
            },
            "intrinsics_source": "color",
            "color_intrinsics": {
                "fx": float(color_intrinsics.fx),
                "fy": float(color_intrinsics.fy),
                "ppx": float(color_intrinsics.ppx),
                "ppy": float(color_intrinsics.ppy),
                "width": int(color_intrinsics.width),
                "height": int(color_intrinsics.height),
            },
            "depth_intrinsics": {
                "fx": float(depth_intrinsics.fx),
                "fy": float(depth_intrinsics.fy),
                "ppx": float(depth_intrinsics.ppx),
                "ppy": float(depth_intrinsics.ppy),
                "width": int(depth_intrinsics.width),
                "height": int(depth_intrinsics.height),
            },
            "depth_scale_m_per_unit": depth_scale_m_per_unit,
        }
    except Exception:
        return None


def export_depth_npz_and_intrinsics(dataset_root: Path) -> None:
    """
    Export captured raw uint16 depth frames to depth.npz for downstream extraction.
    Output layout:
    - {dataset_root}/depth/{camera_key}/{chunk_name}/{file_stem}/depth.npz
    - {dataset_root}/depth/{camera_key}/{chunk_name}/{file_stem}/intrinsics.json
    """
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        print(f"[WARN] dataset videos dir not found: {videos_root}")
        return

    converted_count = 0
    camera_folders = sorted(videos_root.glob("observation.images.*"))
    for camera_folder in camera_folders:
        camera_key = camera_folder.name.replace("observation.images.", "")
        depth_frames = CAPTURED_DEPTH_FRAMES.get(camera_key, [])
        if not depth_frames:
            continue
        metadata = CAPTURED_DEPTH_CAMERA_METADATA.get(camera_key)
        if metadata is None:
            print(f"[WARN] no captured intrinsics metadata for camera key: {camera_key}")
        cursor = 0
        for rgb_video_path in sorted(camera_folder.glob("chunk-*/file-*.mp4")):
            chunk_name = rgb_video_path.parent.name
            file_stem = rgb_video_path.stem
            output_dir = dataset_root / "depth" / camera_key / chunk_name / file_stem
            output_dir.mkdir(parents=True, exist_ok=True)
            output_npz_path = output_dir / "depth.npz"
            output_intrinsics_path = output_dir / "intrinsics.json"

            capture = cv2.VideoCapture(str(rgb_video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open RGB video to infer frame count: {rgb_video_path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

            if frame_count <= 0:
                continue
            remaining = len(depth_frames) - cursor
            take = min(frame_count, max(remaining, 0))
            if take <= 0:
                print(f"[WARN] no remaining depth frames for {camera_key}/{chunk_name}/{file_stem}")
                continue
            depth_stack_mm = np.stack(depth_frames[cursor : cursor + take], axis=0).astype(np.uint16)
            cursor += take
            np.savez_compressed(output_npz_path, depth_mm=depth_stack_mm)
            if metadata is not None:
                output_intrinsics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            converted_count += 1
            print(f"[INFO] wrote {output_npz_path}")
            if metadata is not None:
                print(f"[INFO] wrote {output_intrinsics_path}")

    if converted_count == 0:
        print("[WARN] no depth.npz exported from captured raw depth frames.")
    else:
        print(f"[INFO] exported depth files: {converted_count}")


def build_episode_and_merged_depth_npz(dataset_root: Path) -> None:
    """
    Build episode-wise depth npz and a merged all-depth npz.

    Input (already exported):
    - depth/{camera_key}/chunk-xxx/file-yyy/depth.npz
    - depth/{camera_key}/chunk-xxx/file-yyy/intrinsics.json
    - data/chunk-xxx/file-yyy.parquet (must contain episode_index)

    Output:
    - depth/{camera_key}/episodes/episode-xxxxxx.npz
    - depth/{camera_key}/episodes/manifest.json
    - depth/{camera_key}/depth.npz
    - depth/{camera_key}/depth_index.json
    - depth/{camera_key}/intrinsics.json
    """
    depth_root = dataset_root / "depth"
    data_root = dataset_root / "data"
    if not depth_root.exists() or not data_root.exists():
        return

    for camera_dir in sorted(depth_root.glob("*")):
        if not camera_dir.is_dir():
            continue
        camera_key = camera_dir.name
        depth_files = sorted(camera_dir.glob("chunk-*/file-*/depth.npz"))
        if not depth_files:
            continue

        episodes_root = camera_dir / "episodes"
        if episodes_root.exists():
            # Rebuild fully to avoid duplicate appends across reruns.
            for path in sorted(episodes_root.glob("*")):
                if path.is_file():
                    path.unlink(missing_ok=True)
        episodes_root.mkdir(parents=True, exist_ok=True)

        episode_chunks: dict[int, list[np.ndarray]] = {}
        episode_intrinsics: dict[int, dict[str, Any]] = {}
        file_manifest: list[dict[str, Any]] = []

        for depth_npz_path in depth_files:
            file_dir = depth_npz_path.parent
            chunk_name = file_dir.parent.name
            file_name = file_dir.name
            parquet_path = data_root / chunk_name / f"{file_name}.parquet"
            intrinsics_path = file_dir / "intrinsics.json"

            if not parquet_path.exists():
                print(f"[WARN] missing parquet for depth split: {parquet_path}")
                continue

            try:
                depth_stack = np.load(depth_npz_path)["depth_mm"]
            except Exception as exc:
                print(f"[WARN] failed reading depth npz {depth_npz_path}: {exc}")
                continue

            df = None
            last_parquet_error: Exception | None = None
            for _ in range(5):
                try:
                    df = pd.read_parquet(parquet_path, columns=["episode_index"])
                    last_parquet_error = None
                    break
                except Exception as parquet_error:
                    last_parquet_error = parquet_error
                    time.sleep(0.2)
            if df is None:
                print(f"[WARN] failed reading parquet {parquet_path}: {last_parquet_error}")
                continue
            episode_indices = df["episode_index"].to_numpy()
            n = min(len(depth_stack), len(episode_indices))
            if n <= 0:
                continue
            if len(depth_stack) != len(episode_indices):
                print(
                    f"[WARN] length mismatch for {camera_key}/{chunk_name}/{file_name}: "
                    f"depth={len(depth_stack)} parquet={len(episode_indices)} use={n}"
                )
            depth_stack = depth_stack[:n]
            episode_indices = episode_indices[:n]

            intrinsics_payload: dict[str, Any] | None = None
            if intrinsics_path.exists():
                try:
                    intrinsics_payload = json.loads(intrinsics_path.read_text(encoding="utf-8"))
                except Exception:
                    intrinsics_payload = None

            for episode_id in sorted(set(int(v) for v in episode_indices)):
                mask = episode_indices == episode_id
                if not np.any(mask):
                    continue
                episode_depth = depth_stack[mask]
                episode_chunks.setdefault(episode_id, []).append(episode_depth)
                if intrinsics_payload is not None and episode_id not in episode_intrinsics:
                    episode_intrinsics[episode_id] = intrinsics_payload

                file_manifest.append(
                    {
                        "camera_key": camera_key,
                        "chunk": chunk_name,
                        "file": file_name,
                        "episode_index": episode_id,
                        "frames": int(episode_depth.shape[0]),
                    }
                )

        # Write per-episode depth files.
        episode_summary: list[dict[str, Any]] = []
        for episode_id in sorted(episode_chunks.keys()):
            merged_episode = np.concatenate(episode_chunks[episode_id], axis=0).astype(np.uint16)
            episode_npz_path = episodes_root / f"episode-{episode_id:06d}.npz"
            np.savez_compressed(episode_npz_path, depth_mm=merged_episode)
            episode_summary.append(
                {
                    "episode_index": episode_id,
                    "frames": int(merged_episode.shape[0]),
                    "path": str(episode_npz_path.relative_to(dataset_root)),
                }
            )

        # Write merged all-depth file.
        merged_all_parts: list[np.ndarray] = []
        merged_index: list[dict[str, Any]] = []
        cursor = 0
        for item in episode_summary:
            ep = int(item["episode_index"])
            ep_depth = np.load(episodes_root / f"episode-{ep:06d}.npz")["depth_mm"]
            start = cursor
            end = cursor + int(ep_depth.shape[0])
            merged_all_parts.append(ep_depth)
            merged_index.append({"episode_index": ep, "start": start, "end": end})
            cursor = end

        if merged_all_parts:
            all_depth = np.concatenate(merged_all_parts, axis=0).astype(np.uint16)
            np.savez_compressed(camera_dir / "depth.npz", depth_mm=all_depth)
            (camera_dir / "depth_index.json").write_text(
                json.dumps(
                    {
                        "camera_key": camera_key,
                        "total_frames": int(all_depth.shape[0]),
                        "episodes": merged_index,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            # Store one canonical intrinsics file for this camera.
            if episode_intrinsics:
                first_episode = sorted(episode_intrinsics.keys())[0]
                (camera_dir / "intrinsics.json").write_text(
                    json.dumps(episode_intrinsics[first_episode], indent=2), encoding="utf-8"
                )

        (episodes_root / "manifest.json").write_text(
            json.dumps(
                {
                    "camera_key": camera_key,
                    "episodes": episode_summary,
                    "files": file_manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[INFO] built episode depth + merged depth for {camera_key}: "
            f"{len(episode_summary)} episodes"
        )


def cleanup_chunk_depth_files(dataset_root: Path) -> None:
    """
    Remove per-chunk depth artifacts after merged outputs are built successfully.
    Keep:
    - depth/{camera}/depth.npz
    - depth/{camera}/depth_index.json
    - depth/{camera}/intrinsics.json
    - depth/{camera}/episodes/*
    Remove:
    - depth/{camera}/chunk-*/file-*/depth.npz + intrinsics.json (+ empty dirs)
    """
    depth_root = dataset_root / "depth"
    if not depth_root.exists():
        return

    removed = 0
    for camera_dir in sorted(depth_root.glob("*")):
        if not camera_dir.is_dir():
            continue
        for chunk_dir in sorted(camera_dir.glob("chunk-*")):
            if not chunk_dir.is_dir():
                continue
            for file_dir in sorted(chunk_dir.glob("file-*")):
                if not file_dir.is_dir():
                    continue
                for name in ("depth.npz", "intrinsics.json"):
                    p = file_dir / name
                    if p.exists():
                        p.unlink(missing_ok=True)
                        removed += 1
                # remove empty file-* dir
                try:
                    file_dir.rmdir()
                except OSError:
                    pass
            # remove empty chunk-* dir
            try:
                chunk_dir.rmdir()
            except OSError:
                pass
    if removed:
        print(f"[INFO] cleaned chunk depth artifacts: {removed} files")


def make_robot_from_config_with_depth(config):
    if CAMERA_ONLY_MODE:
        return CameraOnlyRobotWithDepth(config)
    if getattr(config, "type", None) == "so101_follower":
        return SO101FollowerWithDepth(config)
    return official_make_robot_from_config(config)


def camera_only_record_loop(*, robot, events, fps, robot_observation_processor, dataset=None, control_time_s=None, single_task=None, display_data=False, **kwargs):
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import log_rerun_data

    def _pick_minimal_views(observation: dict[str, Any]) -> dict[str, Any]:
        # Keep rerun UI minimal: show all available RGB cameras + one depth view.
        rgb_keys: list[Any] = []
        for key, value in observation.items():
            key_text = str(key).lower()
            if "depth_packed" in key_text:
                continue
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
                if "depth" not in key_text:
                    rgb_keys.append(key)

        minimal: dict[str, Any] = {}
        for key in rgb_keys:
            minimal[f"rgb.{key}"] = observation[key]
        # Prefer d435 depth preview when available.
        preferred_depth_key = None
        if LATEST_DEPTH_PREVIEW:
            for key in LATEST_DEPTH_PREVIEW.keys():
                if "d435" in str(key).lower():
                    preferred_depth_key = key
                    break
            if preferred_depth_key is None:
                preferred_depth_key = next(iter(LATEST_DEPTH_PREVIEW.keys()))
        if preferred_depth_key is not None:
            minimal["depth"] = LATEST_DEPTH_PREVIEW[preferred_depth_key]
        return minimal

    if dataset is not None:
        DEPTH_EPISODE_SNAPSHOTS.append(_snapshot_depth_lengths())
    _set_depth_capture_enabled(dataset is not None)
    try:
        timestamp = 0.0
        start_episode_time = time.perf_counter()
        while timestamp < float(control_time_s):
            start_loop_time = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            observation = robot.get_observation()
            processed_observation = robot_observation_processor(observation)

            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, processed_observation, prefix=OBS_STR)
                dataset.add_frame({**observation_frame, "task": single_task})

            if display_data:
                display_observation = _pick_minimal_views(processed_observation)
                log_rerun_data(observation=display_observation, action={})

            dt_s = time.perf_counter() - start_loop_time
            precise_sleep(1 / fps - dt_s)
            timestamp = time.perf_counter() - start_episode_time
    finally:
        _set_depth_capture_enabled(True)


def rawdata_record_main() -> None:
    global CAMERA_ONLY_MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--export-depth-npz", action="store_true", default=True)
    parser.add_argument("--no-export-depth-npz", action="store_true")
    parsed_arguments, _ = parser.parse_known_args()

    export_depth_npz_enabled = bool(parsed_arguments.export_depth_npz and not parsed_arguments.no_export_depth_npz)
    interrupted = False
    dataset_root_value_for_export = _extract_cli_value(sys.argv[1:], "--dataset.root")
    dataset_root_for_export = (
        Path(dataset_root_value_for_export).expanduser().resolve() if dataset_root_value_for_export else None
    )

    original_record_loop = official_record.record_loop

    def depth_gated_record_loop(*args, **kwargs):
        dataset = kwargs.get("dataset")
        if dataset is not None:
            DEPTH_EPISODE_SNAPSHOTS.append(_snapshot_depth_lengths())
        _set_depth_capture_enabled(dataset is not None)
        try:
            return original_record_loop(*args, **kwargs)
        finally:
            _set_depth_capture_enabled(True)

    official_record.record_loop = depth_gated_record_loop

    if parsed_arguments.camera_only:
        CAMERA_ONLY_MODE = True
        filtered_arguments = [
            argument
            for argument in sys.argv[1:]
            if argument not in {"--camera-only", "--export-depth-npz", "--no-export-depth-npz"}
        ]
        if not _has_cli_key(filtered_arguments, "--robot.type"):
            filtered_arguments.extend(["--robot.type=so101_follower"])
        if not _has_cli_key(filtered_arguments, "--robot.id"):
            filtered_arguments.extend(["--robot.id=camera_only"])
        if not _has_cli_key(filtered_arguments, "--robot.port"):
            filtered_arguments.extend(["--robot.port=/dev/null"])
        if not _has_cli_key(filtered_arguments, "--dataset.repo_id"):
            filtered_arguments.extend(["--dataset.repo_id=local/camera_only"])
        has_teleop = _has_cli_prefix(filtered_arguments, "--teleop.")
        has_policy = _has_cli_prefix(filtered_arguments, "--policy.")
        if not has_teleop and not has_policy:
            filtered_arguments.extend(["--teleop.type=keyboard"])
        sys.argv = [sys.argv[0]] + filtered_arguments
        official_record.record_loop = camera_only_record_loop

    if not parsed_arguments.camera_only:
        filtered_arguments = [
            argument
            for argument in sys.argv[1:]
            if argument not in {"--export-depth-npz", "--no-export-depth-npz"}
        ]
        sys.argv = [sys.argv[0]] + filtered_arguments

    # Monkey-patch only in this process.
    official_record.make_robot_from_config = make_robot_from_config_with_depth
    original_init_rerun = official_record.init_rerun
    dataset_class = official_record.LeRobotDataset
    original_clear_episode_buffer = dataset_class.clear_episode_buffer
    original_save_episode = dataset_class.save_episode

    def _clear_episode_buffer_with_depth_rollback(self, *args, **kwargs):
        if DEPTH_EPISODE_SNAPSHOTS:
            snapshot = DEPTH_EPISODE_SNAPSHOTS.pop()
            _restore_depth_lengths(snapshot)
        return original_clear_episode_buffer(self, *args, **kwargs)

    def _save_episode_with_depth_commit(self):
        snapshot = _snapshot_depth_lengths()
        if DEPTH_EPISODE_SNAPSHOTS:
            snapshot = DEPTH_EPISODE_SNAPSHOTS.pop()
        result = original_save_episode(self)
        # Commit depth directly as one episode file from in-memory snapshot.
        if export_depth_npz_enabled:
            try:
                root = getattr(self, "root", None) or dataset_root_for_export
                if root is not None:
                    _write_episode_depth_from_snapshot(dataset_root=Path(root), snapshot=snapshot)
            except Exception as export_error:
                print(f"[WARN] per-episode depth export failed: {export_error}")
        return result

    dataset_class.clear_episode_buffer = _clear_episode_buffer_with_depth_rollback
    dataset_class.save_episode = _save_episode_with_depth_commit

    def _init_rerun_fresh(session_name: str = "recording"):
        # Force a fresh rerun session name each run to avoid reusing stale blueprint layouts.
        fresh_name = f"recording_minimal_{int(time.time())}"
        return original_init_rerun(session_name=fresh_name)

    official_record.init_rerun = _init_rerun_fresh
    register_third_party_plugins()
    try:
        try:
            official_record.record()
        except KeyboardInterrupt:
            interrupted = True
            print("[WARN] Recording interrupted by user (Ctrl+C).")
    finally:
        # Keep final export as a second safety net.
        if export_depth_npz_enabled:
            if dataset_root_for_export is not None:
                if interrupted:
                    print("[WARN] Skip merged depth build on interrupted run.")
                else:
                    _build_merged_depth_from_episodes(dataset_root=dataset_root_for_export)
            else:
                print("[WARN] --dataset.root not found; skip final depth.npz export.")


if __name__ == "__main__":
    rawdata_record_main()
