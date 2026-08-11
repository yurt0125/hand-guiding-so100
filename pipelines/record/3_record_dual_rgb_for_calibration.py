#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def open_camera(source: Any, width: int, height: int, fps: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    camera.set(cv2.CAP_PROP_FPS, float(fps))
    return camera


def ensure_video_writer(path: Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def probe_cameras(max_index: int) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            continue
        ok, frame = capture.read()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        discovered.append(
            {
                "index": index,
                "read_ok": bool(ok),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_shape": None if frame is None else tuple(frame.shape),
            }
        )
        capture.release()
    return discovered


def parse_source(index_value: int | None, source_value: str | None) -> Any:
    if source_value is not None and source_value != "":
        return source_value
    if index_value is None:
        raise ValueError("Missing camera source. Provide either index or source path.")
    return int(index_value)


def main():
    parser = argparse.ArgumentParser(description="Record dual RGB videos for camera calibration.")
    parser.add_argument("--output-dir", default="calibration/inputs/videos", help="Directory for output files.")
    parser.add_argument("--d435-index", type=int, default=0, help="OpenCV camera index for D435 RGB.")
    parser.add_argument("--top-index", type=int, default=1, help="OpenCV camera index for top RGB.")
    parser.add_argument("--d435-source", default=None, help="Optional camera source path for D435, e.g. /dev/video2.")
    parser.add_argument("--top-source", default=None, help="Optional camera source path for top camera.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    parser.add_argument("--duration-s", type=float, default=20.0, help="Max recording duration in seconds.")
    parser.add_argument("--d435-name", default="d435_calib", help="Base filename for D435 outputs.")
    parser.add_argument("--top-name", default="top_calib", help="Base filename for top outputs.")
    parser.add_argument("--show", action="store_true", help="Show live preview window.")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices then exit.")
    parser.add_argument("--probe-max-index", type=int, default=10, help="Max camera index for probing.")
    args = parser.parse_args()

    if args.list_cameras:
        discovered = probe_cameras(max_index=int(args.probe_max_index))
        print(f"[INFO] Probed indices: 0..{int(args.probe_max_index)}")
        if len(discovered) == 0:
            print("[WARN] No cameras discovered.")
            return
        for item in discovered:
            print(
                f"[INFO] index={item['index']} read_ok={item['read_ok']} "
                f"size={item['width']}x{item['height']} fps={item['fps']:.2f} shape={item['frame_shape']}"
            )
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d435_video_path = output_dir / f"{args.d435_name}.mp4"
    top_video_path = output_dir / f"{args.top_name}.mp4"
    d435_timestamps_path = output_dir / f"{args.d435_name}_timestamps.npy"
    top_timestamps_path = output_dir / f"{args.top_name}_timestamps.npy"
    d435_meta_path = output_dir / f"{args.d435_name}_frame_meta.jsonl"
    top_meta_path = output_dir / f"{args.top_name}_frame_meta.jsonl"

    d435_source = parse_source(args.d435_index, args.d435_source)
    top_source = parse_source(args.top_index, args.top_source)
    d435_capture = open_camera(d435_source, args.width, args.height, args.fps)
    top_capture = open_camera(top_source, args.width, args.height, args.fps)
    d435_writer = ensure_video_writer(d435_video_path, args.fps, args.width, args.height)
    top_writer = ensure_video_writer(top_video_path, args.fps, args.width, args.height)

    d435_timestamps: list[float] = []
    top_timestamps: list[float] = []

    print("[INFO] Dual camera recorder ready.")
    print(f"[INFO] D435 source: {d435_source}")
    print(f"[INFO] Top source: {top_source}")
    print("[INFO] Press SPACE to start recording, Q or ESC to stop early.")

    started = False
    start_time = 0.0
    d435_frame_index = 0
    top_frame_index = 0

    d435_meta_handle = d435_meta_path.open("w", encoding="utf-8")
    top_meta_handle = top_meta_path.open("w", encoding="utf-8")

    try:
        while True:
            d435_ok, d435_frame = d435_capture.read()
            top_ok, top_frame = top_capture.read()
            if not d435_ok or not top_ok:
                print("[WARN] Frame grab failed. Stopping.")
                break

            now = time.time()
            if started:
                d435_writer.write(d435_frame)
                top_writer.write(top_frame)

                d435_timestamps.append(now)
                top_timestamps.append(now)

                d435_meta_handle.write(
                    json.dumps({"frame_index": d435_frame_index, "timestamp_s": now}, ensure_ascii=False) + "\n"
                )
                top_meta_handle.write(
                    json.dumps({"frame_index": top_frame_index, "timestamp_s": now}, ensure_ascii=False) + "\n"
                )
                d435_frame_index += 1
                top_frame_index += 1

            if args.show:
                left = cv2.resize(d435_frame, (args.width, args.height))
                right = cv2.resize(top_frame, (args.width, args.height))
                canvas = np.hstack([left, right])
                status = "RECORDING" if started else "IDLE"
                elapsed = now - start_time if started else 0.0
                cv2.putText(
                    canvas,
                    f"{status}  elapsed={elapsed:.1f}s  frames={d435_frame_index}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0) if started else (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("dual_calibration_recorder", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and not started:
                started = True
                start_time = time.time()
                print("[INFO] Recording started.")
            elif key in (ord("q"), 27):
                print("[INFO] Recording stopped by user.")
                break

            if started and (time.time() - start_time) >= float(args.duration_s):
                print("[INFO] Recording duration reached.")
                break
    finally:
        d435_meta_handle.close()
        top_meta_handle.close()
        d435_writer.release()
        top_writer.release()
        d435_capture.release()
        top_capture.release()
        if args.show:
            cv2.destroyAllWindows()

    np.save(str(d435_timestamps_path), np.asarray(d435_timestamps, dtype=float))
    np.save(str(top_timestamps_path), np.asarray(top_timestamps, dtype=float))

    summary = {
        "d435_video_path": str(d435_video_path),
        "top_video_path": str(top_video_path),
        "d435_timestamps_path": str(d435_timestamps_path),
        "top_timestamps_path": str(top_timestamps_path),
        "d435_frame_meta_path": str(d435_meta_path),
        "top_frame_meta_path": str(top_meta_path),
        "frames_recorded": int(min(d435_frame_index, top_frame_index)),
        "fps_target": int(args.fps),
        "width": int(args.width),
        "height": int(args.height),
    }
    summary_path = output_dir / "dual_record_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[INFO] D435 video: {d435_video_path}")
    print(f"[INFO] Top video: {top_video_path}")
    print(f"[INFO] D435 timestamps: {d435_timestamps_path}")
    print(f"[INFO] Top timestamps: {top_timestamps_path}")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
