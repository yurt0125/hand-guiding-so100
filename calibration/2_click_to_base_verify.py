#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click a D435 pixel and print its base-frame coordinates."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--patch-radius",
        type=int,
        default=2,
        help="Median depth patch radius (pixels).",
    )
    parser.add_argument(
        "--transform-json",
        default="calibration/outputs/cam2base_latest.json",
        help="Camera-to-base transform JSON containing T_cam2base.",
    )
    return parser.parse_args()


def load_cam2base(path: Path) -> np.ndarray:
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


def get_depth_median_at_pixel(depth_frame, u: int, v: int, patch_radius: int = 1) -> float:
    values = []
    for dy in range(-patch_radius, patch_radius + 1):
        for dx in range(-patch_radius, patch_radius + 1):
            uu = u + dx
            vv = v + dy
            if uu < 0 or vv < 0:
                continue
            depth_value = float(depth_frame.get_distance(int(uu), int(vv)))
            if depth_value > 0:
                values.append(depth_value)
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def main() -> None:
    arguments = parse_arguments()
    transform = load_cam2base(Path(arguments.transform_json).expanduser().resolve())

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, arguments.width, arguments.height, rs.format.z16, arguments.fps)
    config.enable_stream(rs.stream.color, arguments.width, arguments.height, rs.format.bgr8, arguments.fps)
    _profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrinsics = None

    clicked_pixel: Optional[Tuple[int, int]] = None

    def on_mouse(event, x, y, _flags, _param):
        nonlocal clicked_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixel = (int(x), int(y))

    cv2.namedWindow("D435 Verify", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("D435 Verify", on_mouse)
    print("左键点击要验证的点；按 Q/ESC 退出。")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            if intrinsics is None:
                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()

            image = np.asanyarray(color_frame.get_data())
            if clicked_pixel is not None:
                cv2.circle(image, clicked_pixel, 6, (0, 0, 255), -1)
            cv2.imshow("D435 Verify", image)

            if clicked_pixel is not None:
                u, v = clicked_pixel
                clicked_pixel = None

                depth_m = get_depth_median_at_pixel(depth_frame, u, v, patch_radius=arguments.patch_radius)
                if depth_m <= 0:
                    print(f"[WARN] ({u},{v}) 深度无效。")
                    continue

                cam_point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth_m)
                cam_h = np.array([cam_point[0], cam_point[1], cam_point[2], 1.0], dtype=float)
                base_point = transform @ cam_h
                print(
                    f"[CLICK] pixel=({u},{v}) depth={depth_m:.4f}m "
                    f"-> base=({base_point[0]:.4f}, {base_point[1]:.4f}, {base_point[2]:.4f})"
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
