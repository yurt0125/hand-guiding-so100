#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from pathlib import Path

# Keep numerical backends from oversubscribing CPU while the robot command
# loop and RealSense UI run together.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import cv2
import mujoco
import numpy as np
import pyrealsense2 as rs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.keyboard_mirror_app import KeyboardController, SO100TeleopViewer
from common.common_robot import SCENE_XML_PATH, SO100Hardware


# Keep the same click behavior as old 1_d435_click.py
clicked_pixel: tuple[int, int] | None = None
new_click = False
JAW_TO_IK_CORRECTION = np.array([-0.0202, 0.01, 0.023], dtype=float)


def mouse_callback(event, x, y, flags, param):
    global clicked_pixel, new_click
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (int(x), int(y))
        new_click = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D435 click calibration + keyboard robot control (minimal mode)."
    )
    parser.add_argument("--robot-port", default="/dev/ttyACM0", help="SO100 USB port")
    parser.add_argument(
        "--output-csv",
        default="calibration/points/dual_camera_points.csv",
        help="Output correspondence CSV",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--rs-serial",
        default=None,
        help="Optional RealSense serial number. Use this when multiple cameras are connected.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=30.0,
        help="Robot command rate during keyboard calibration. Lower this if motion stutters.",
    )
    parser.add_argument(
        "--profile-control",
        action="store_true",
        help="Print robot control loop timing diagnostics once per second.",
    )
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument(
        "--output-json",
        default="calibration/outputs/cam2base_latest.json",
        help="Output cam2base json path.",
    )
    return parser.parse_args()


def rigid_transform_3d(camera_points: np.ndarray, robot_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera_points.shape != robot_points.shape:
        raise ValueError(f"Shape mismatch: {camera_points.shape} vs {robot_points.shape}")
    camera_centroid = np.mean(camera_points, axis=0)
    robot_centroid = np.mean(robot_points, axis=0)

    camera_centered = camera_points - camera_centroid
    robot_centered = robot_points - robot_centroid
    covariance_matrix = camera_centered.T @ robot_centered

    left_vectors, _, right_vectors_t = np.linalg.svd(covariance_matrix)
    rotation_matrix = right_vectors_t.T @ left_vectors.T
    if np.linalg.det(rotation_matrix) < 0:
        right_vectors_t[2, :] *= -1
        rotation_matrix = right_vectors_t.T @ left_vectors.T
    translation_vector = robot_centroid - rotation_matrix @ camera_centroid

    transform_matrix = np.eye(4, dtype=float)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector
    return transform_matrix, rotation_matrix, translation_vector


def solve_and_save_cam2base(rows: list[dict[str, float]], output_json_path: Path, source_csv_path: Path) -> None:
    camera_points = np.asarray(
        [[row["d435_cam_x"], row["d435_cam_y"], row["d435_cam_z"]] for row in rows],
        dtype=float,
    )
    robot_points = np.asarray(
        [[row["robot_x"], row["robot_y"], row["robot_z"]] for row in rows],
        dtype=float,
    )
    transform_matrix, rotation_matrix, translation_vector = rigid_transform_3d(camera_points, robot_points)
    predicted_robot = (rotation_matrix @ camera_points.T).T + translation_vector
    point_errors_mm = np.linalg.norm(predicted_robot - robot_points, axis=1) * 1000.0
    rmse_mm = float(np.sqrt(np.mean(np.square(point_errors_mm))))

    payload = {
        "T_cam2base": transform_matrix.tolist(),
        "T_base_camera": transform_matrix.tolist(),
        "T_camera_base": np.linalg.inv(transform_matrix).tolist(),
        "metadata": {
            "source_points_file": str(source_csv_path),
            "num_points": int(len(rows)),
            "rmse_mm": rmse_mm,
            "max_error_mm": float(np.max(point_errors_mm)),
            "mean_error_mm": float(np.mean(point_errors_mm)),
            "method": "svd_rigid_transform_3d",
        },
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("\n==========================================")
    print("✅ 终极 4x4 手眼转换矩阵 (Camera -> Robot):")
    print("T_cam2base = np.array([")
    for row in transform_matrix:
        print(f"    [{row[0]:>8.4f}, {row[1]:>8.4f}, {row[2]:>8.4f}, {row[3]:>8.4f}],")
    print("])")
    print("==========================================")
    print(f"🎯 标定误差 (RMSE): {rmse_mm:.2f} 毫米 (mm)")
    print("==========================================")
    print(f"[INFO] Saved cam2base json: {output_json_path}")


def main() -> None:
    global clicked_pixel, new_click
    arguments = parse_arguments()
    cv2.setNumThreads(1)

    output_csv_path = Path(arguments.output_csv).expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path = Path(arguments.output_json).expanduser().resolve()

    # Robot control: reuse proven keyboard controller path
    robot_hardware = SO100Hardware(port=arguments.robot_port, use_degrees=True)
    robot_hardware.connect()
    keyboard_controller = KeyboardController()
    control_model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
    control_data = mujoco.MjData(control_model)
    keyboard_viewer = SO100TeleopViewer(
        model=control_model,
        data=control_data,
        controller=keyboard_controller,
        real_robot=robot_hardware.robot,
        enable_viewer=False,
        enable_status_output=False,
        status_use_carriage_return=False,
        control_hz=arguments.control_hz,
        profile_control=arguments.profile_control,
    )
    control_thread = threading.Thread(target=keyboard_viewer.run_loop, daemon=True)
    control_thread.start()

    # D435 stream: same as old script
    pipeline = rs.pipeline()
    config = rs.config()
    if arguments.rs_serial:
        config.enable_device(str(arguments.rs_serial))
    config.enable_stream(rs.stream.depth, arguments.width, arguments.height, rs.format.z16, arguments.fps)
    config.enable_stream(rs.stream.color, arguments.width, arguments.height, rs.format.bgr8, arguments.fps)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    cv2.namedWindow("D435 Color Feed", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("D435 Color Feed", mouse_callback)

    print("=" * 50)
    print("📷 D435 打点 + 🤖 机械臂键盘控制（精简版）")
    print("=" * 50)
    print("左键点击: 读取相机3D")
    print("G: 记录一组点 (camera_xyz + robot_xyz)")
    print("P: 保存 CSV")
    print("B: 清除当前点击")
    print("X/ESC: 退出")

    latest_depth_m: float | None = None
    latest_camera_point: np.ndarray | None = None
    rows: list[dict[str, float]] = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()
            ik_xyz = np.array(
                [keyboard_controller.x, keyboard_controller.y, keyboard_controller.z],
                dtype=float,
            )
            jaw_xyz = ik_xyz + JAW_TO_IK_CORRECTION

            if new_click and clicked_pixel is not None:
                u, v = clicked_pixel
                depth_value = float(aligned_depth_frame.get_distance(u, v))
                if depth_value > 0:
                    camera_3d_point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_value)
                    latest_depth_m = depth_value
                    latest_camera_point = np.asarray(camera_3d_point, dtype=float)
                    print(f"\n🎯 [鼠标点击] 像素: ({u}, {v}) | 深度: {depth_value:.3f} 米")
                    print(
                        f"📦 [相机坐标系 3D]: X={latest_camera_point[0]:.4f}, "
                        f"Y={latest_camera_point[1]:.4f}, Z={latest_camera_point[2]:.4f}"
                    )
                else:
                    latest_depth_m = None
                    latest_camera_point = None
                    print(f"\n⚠️ 无法获取像素 ({u},{v}) 的深度值。")
                new_click = False

            if clicked_pixel is not None:
                cv2.circle(display_image, clicked_pixel, 5, (0, 0, 255), -1)
                cv2.putText(
                    display_image,
                    "Target",
                    (clicked_pixel[0] + 10, clicked_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                display_image,
                f"pairs: {len(rows)} / min:{arguments.min_pairs}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_image,
                (
                    "robot(jaw) "
                    f"x={jaw_xyz[0]:+.3f} "
                    f"y={jaw_xyz[1]:+.3f} "
                    f"z={jaw_xyz[2]:+.3f}"
                ),
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 255, 80),
                2,
            )
            cv2.imshow("D435 Color Feed", display_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("x"), ord("X"), 27):
                break
            if key == ord("b"):
                clicked_pixel = None
                latest_depth_m = None
                latest_camera_point = None
                print("[INFO] Cleared current click.")
            if key == ord("g"):
                if clicked_pixel is None or latest_camera_point is None or latest_depth_m is None:
                    print("[WARN] 请先左键点击并确保深度有效。")
                    continue
                ik_xyz = np.array(
                    [keyboard_controller.x, keyboard_controller.y, keyboard_controller.z],
                    dtype=float,
                )
                robot_xyz = ik_xyz + JAW_TO_IK_CORRECTION
                row = {
                    "d435_u": float(clicked_pixel[0]),
                    "d435_v": float(clicked_pixel[1]),
                    "d435_depth_m": float(latest_depth_m),
                    "d435_cam_x": float(latest_camera_point[0]),
                    "d435_cam_y": float(latest_camera_point[1]),
                    "d435_cam_z": float(latest_camera_point[2]),
                    "robot_x": float(robot_xyz[0]),
                    "robot_y": float(robot_xyz[1]),
                    "robot_z": float(robot_xyz[2]),
                }
                rows.append(row)
                print(
                    "🤖 [机械臂坐标系 3D / JAW]: "
                    f"x={robot_xyz[0]:.3f}, y={robot_xyz[1]:.3f}, z={robot_xyz[2]:.3f}"
                )
                print(f"[INFO] Pair #{len(rows)} captured")
                clicked_pixel = None
                latest_depth_m = None
                latest_camera_point = None
            if key == ord("p"):
                if len(rows) < arguments.min_pairs:
                    print(f"[WARN] Need at least {arguments.min_pairs} pairs, current {len(rows)}.")
                    continue
                with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"[INFO] Saved {len(rows)} pairs to: {output_csv_path}")
                solve_and_save_cam2base(rows, output_json_path, output_csv_path)
                break

    finally:
        keyboard_viewer.stop_requested = True
        control_thread.join(timeout=1.0)
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        keyboard_controller.cleanup()
        print("")
        robot_hardware.disconnect()


if __name__ == "__main__":
    main()
