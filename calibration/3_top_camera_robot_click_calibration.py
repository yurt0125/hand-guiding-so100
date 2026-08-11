#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from pathlib import Path

import cv2
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.keyboard_mirror_app import KeyboardController, SO100TeleopViewer
from common.common_robot import SCENE_XML_PATH, SO100Hardware


clicked_pixel: tuple[int, int] | None = None
new_click = False


def mouse_callback(event, x, y, flags, param):
    global clicked_pixel, new_click
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (int(x), int(y))
        new_click = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top RGB click calibration + keyboard robot control (minimal mode)."
    )
    parser.add_argument("--top-source", required=True, help="Top camera source, e.g. /dev/video4")
    parser.add_argument("--robot-port", default="/dev/ttyACM0", help="SO100 USB port")
    parser.add_argument(
        "--top-intrinsics-json",
        default="calibration/outputs/top_intrinsics.json",
        help="Top camera intrinsics json path",
    )
    parser.add_argument(
        "--output-csv",
        default="calibration/points/top_camera_points.csv",
        help="Output correspondence CSV",
    )
    parser.add_argument(
        "--output-json",
        default="calibration/outputs/T_top_to_base.json",
        help="Output top->base json path",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-pairs", type=int, default=6)
    return parser.parse_args()


def load_intrinsics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = payload["intrinsics"]
    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "ppx": float(intrinsics["ppx"]),
        "ppy": float(intrinsics["ppy"]),
    }


def solve_and_save_top_to_base(
    rows: list[dict[str, float]],
    intrinsics: dict[str, float],
    output_json_path: Path,
    source_csv_path: Path,
) -> None:
    object_points_base = np.asarray(
        [[row["robot_x"], row["robot_y"], row["robot_z"]] for row in rows],
        dtype=np.float64,
    )
    image_points = np.asarray(
        [[row["top_u"], row["top_v"]] for row in rows],
        dtype=np.float64,
    )

    camera_matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((5, 1), dtype=np.float64)

    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        objectPoints=object_points_base.astype(np.float32),
        imagePoints=image_points.astype(np.float32),
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
        reprojectionError=3.0,
        iterationsCount=300,
        confidence=0.995,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=object_points_base.astype(np.float32),
            imagePoints=image_points.astype(np.float32),
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None
    if not success:
        raise RuntimeError("Top camera solvePnP failed.")

    rotation_base_to_cam, _ = cv2.Rodrigues(rotation_vector)
    translation_base_to_cam = translation_vector.reshape(3)

    transform_base_to_cam = np.eye(4, dtype=float)
    transform_base_to_cam[:3, :3] = rotation_base_to_cam
    transform_base_to_cam[:3, 3] = translation_base_to_cam
    transform_cam_to_base = np.linalg.inv(transform_base_to_cam)

    projected, _ = cv2.projectPoints(
        object_points_base.astype(np.float32),
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    reprojection_errors = np.linalg.norm(projected - image_points, axis=1)
    reprojection_rmse = float(np.sqrt(np.mean(np.square(reprojection_errors))))

    payload = {
        "T_top_to_base": transform_cam_to_base.tolist(),
        "T_base_to_top": transform_base_to_cam.tolist(),
        "metadata": {
            "source_points_file": str(source_csv_path),
            "num_points": int(len(rows)),
            "num_inliers": 0 if inliers is None else int(len(inliers)),
            "reprojection_rmse_px": reprojection_rmse,
            "method": "solvePnPRansac+solvePnP_fallback",
            "intrinsics": intrinsics,
        },
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n==========================================")
    print("✅ Top 相机到机械臂基座变换矩阵:")
    print("T_top_to_base = np.array([")
    for row in transform_cam_to_base:
        print(f"    [{row[0]:>8.4f}, {row[1]:>8.4f}, {row[2]:>8.4f}, {row[3]:>8.4f}],")
    print("])")
    print("==========================================")
    print(f"🎯 重投影误差 (RMSE): {reprojection_rmse:.3f} 像素")
    print("==========================================")
    print(f"[INFO] Saved top->base json: {output_json_path}")


def main() -> None:
    global clicked_pixel, new_click
    arguments = parse_arguments()

    intrinsics = load_intrinsics(Path(arguments.top_intrinsics_json).expanduser().resolve())
    output_csv_path = Path(arguments.output_csv).expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path = Path(arguments.output_json).expanduser().resolve()

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
    )
    control_thread = threading.Thread(target=keyboard_viewer.run_loop, daemon=True)
    control_thread.start()

    top_capture = cv2.VideoCapture(arguments.top_source)
    if not top_capture.isOpened():
        raise RuntimeError(f"Failed to open top camera source: {arguments.top_source}")
    top_capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(arguments.width))
    top_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(arguments.height))
    top_capture.set(cv2.CAP_PROP_FPS, float(arguments.fps))

    cv2.namedWindow("Top Color Feed", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Top Color Feed", mouse_callback)

    print("=" * 50)
    print("📷 Top RGB 打点 + 🤖 机械臂键盘控制（精简版）")
    print("=" * 50)
    print("左键点击: 记录 Top 像素")
    print("G: 记录一组点 (top_uv + robot_xyz)")
    print("P: 保存 CSV 并计算 T_top_to_base")
    print("B: 清除当前点击")
    print("Q/ESC: 退出")

    rows: list[dict[str, float]] = []

    try:
        while True:
            ok, frame = top_capture.read()
            if not ok:
                continue
            display_image = frame.copy()

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
            cv2.imshow("Top Color Feed", display_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                clicked_pixel = None
                new_click = False
                print("[INFO] Cleared current click.")
            if key == ord("g"):
                if clicked_pixel is None:
                    print("[WARN] 请先左键点击 top 图像。")
                    continue
                robot_xyz = np.array(
                    [keyboard_controller.x, keyboard_controller.y, keyboard_controller.z],
                    dtype=float,
                )
                row = {
                    "top_u": float(clicked_pixel[0]),
                    "top_v": float(clicked_pixel[1]),
                    "robot_x": float(robot_xyz[0]),
                    "robot_y": float(robot_xyz[1]),
                    "robot_z": float(robot_xyz[2]),
                }
                rows.append(row)
                print(f"🎯 [Top 像素] u={clicked_pixel[0]}, v={clicked_pixel[1]}")
                print(
                    f"🤖 [机械臂坐标系 3D]: x={robot_xyz[0]:.3f}, y={robot_xyz[1]:.3f}, z={robot_xyz[2]:.3f}"
                )
                print(f"[INFO] Pair #{len(rows)} captured")
                clicked_pixel = None
                new_click = False
            if key == ord("p"):
                if len(rows) < arguments.min_pairs:
                    print(f"[WARN] Need at least {arguments.min_pairs} pairs, current {len(rows)}.")
                    continue
                with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"[INFO] Saved {len(rows)} pairs to: {output_csv_path}")
                solve_and_save_top_to_base(rows, intrinsics, output_json_path, output_csv_path)
                break

    finally:
        keyboard_viewer.stop_requested = True
        control_thread.join(timeout=1.0)
        top_capture.release()
        cv2.destroyAllWindows()
        keyboard_controller.cleanup()
        print("")
        robot_hardware.disconnect()


if __name__ == "__main__":
    main()

