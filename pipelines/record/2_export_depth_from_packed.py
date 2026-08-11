#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def decode_packed_depth_video_to_uint16(depth_packed_video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(depth_packed_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open depth packed video: {depth_packed_video_path}")

    frames: list[np.ndarray] = []
    high_channel_index: int | None = None
    low_channel_index: int | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Unexpected depth packed frame shape: {frame.shape}")

            if high_channel_index is None or low_channel_index is None:
                channel_means = frame.reshape(-1, 3).mean(axis=0)
                zero_channel_index = int(np.argmin(channel_means))
                remaining_indices = [index for index in [0, 1, 2] if index != zero_channel_index]
                first_index, second_index = remaining_indices
                if channel_means[first_index] <= channel_means[second_index]:
                    high_channel_index = first_index
                    low_channel_index = second_index
                else:
                    high_channel_index = second_index
                    low_channel_index = first_index

            high_channel = frame[:, :, int(high_channel_index)].astype(np.uint16)
            low_channel = frame[:, :, int(low_channel_index)].astype(np.uint16)
            depth_mm = (high_channel << 8) | low_channel
            frames.append(depth_mm)
    finally:
        capture.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from {depth_packed_video_path}")
    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export depth (depth.npz + intrinsics.json) from depth_packed mp4.")
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root, e.g. data/raw")
    parser.add_argument(
        "--intrinsics-json",
        required=True,
        help="D435 intrinsics json containing intrinsics + depth_scale_m_per_unit",
    )
    parser.add_argument("--camera-key", default="d435", help="Camera key, default: d435")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    intrinsics_json_path = Path(args.intrinsics_json).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if not intrinsics_json_path.exists():
        raise FileNotFoundError(f"intrinsics json not found: {intrinsics_json_path}")

    intrinsics_payload = json.loads(intrinsics_json_path.read_text(encoding="utf-8"))

    packed_root = dataset_root / "videos" / f"observation.images.{args.camera_key}.depth_packed"
    if not packed_root.exists():
        raise FileNotFoundError(f"depth packed root not found: {packed_root}")

    video_paths = sorted(packed_root.glob("chunk-*/file-*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"no depth packed videos found under: {packed_root}")

    converted_count = 0
    for video_path in video_paths:
        chunk_name = video_path.parent.name
        file_stem = video_path.stem
        output_dir = dataset_root / "depth" / args.camera_key / chunk_name / file_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        depth_mm_stack = decode_packed_depth_video_to_uint16(video_path)
        np.savez_compressed(output_dir / "depth.npz", depth_mm=depth_mm_stack)
        (output_dir / "intrinsics.json").write_text(
            json.dumps(intrinsics_payload, indent=2),
            encoding="utf-8",
        )

        converted_count += 1
        print(f"[INFO] exported: {output_dir}")

    print(f"[INFO] converted files: {converted_count}")


if __name__ == "__main__":
    main()
