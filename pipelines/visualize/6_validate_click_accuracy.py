#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from common.common_control import CommonRunner
from common.common_robot import EndEffectorTarget, SO100Hardware, SO100Kinematics
from common.config import (
    DEFAULT_CONTROL_GAIN,
    DEFAULT_FILTER_BACKEND,
    DEFAULT_USB_PORT,
    VISION_WORKSPACE,
    load_camera_to_base_transform,
)


class D435Manager:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()

    def stop(self):
        self.pipeline.stop()


@dataclass
class ClickMeasurement:
    index: int
    pixel_u: int
    pixel_v: int
    depth_m: float
    predicted_base_x: float
    predicted_base_y: float
    predicted_base_z: float
    actual_base_x: float
    actual_base_y: float
    actual_base_z: float
    error_x_m: float
    error_y_m: float
    error_z_m: float
    error_norm_m: float
    timestamp_s: float


class VisionClickValidator:
    def __init__(
        self,
        runner: CommonRunner,
        init_target: EndEffectorTarget,
        camera_to_base_transform: np.ndarray,
        settle_steps: int,
        settle_sleep_s: float,
        approach_height_m: float,
    ):
        self.runner = runner
        self.target = EndEffectorTarget(**init_target.__dict__)
        self.camera_to_base_transform = camera_to_base_transform
        self.settle_steps = int(settle_steps)
        self.settle_sleep_s = float(settle_sleep_s)
        self.approach_height_m = float(approach_height_m)
        self.clicked_pixel: Optional[Tuple[int, int]] = None
        self.measurements: list[ClickMeasurement] = []
        self.last_error_mm: Optional[float] = None

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_pixel = (int(x), int(y))

    def _build_target_from_pixel(
        self, depth_frame, intrinsics
    ) -> Optional[tuple[EndEffectorTarget, int, int, float, float, float, float]]:
        if self.clicked_pixel is None:
            return None

        pixel_u, pixel_v = self.clicked_pixel
        self.clicked_pixel = None

        depth_m = float(depth_frame.get_distance(pixel_u, pixel_v))
        if depth_m <= 0:
            return None

        cam_point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_u, pixel_v], depth_m)
        camera_homogeneous = np.array([cam_point[0], cam_point[1], cam_point[2], 1.0], dtype=float)
        base_point = self.camera_to_base_transform @ camera_homogeneous

        target = EndEffectorTarget(
            x=float(base_point[0]),
            y=float(base_point[1]),
            z=float(base_point[2] + self.approach_height_m),
            tool_axis_x=float(self.target.tool_axis_x),
            tool_axis_y=float(self.target.tool_axis_y),
            tool_axis_z=float(self.target.tool_axis_z),
            gripper=float(self.target.gripper),
            valid=True,
            source="vision_click_validate",
            timestamp=time.time(),
        )
        self.target = EndEffectorTarget(**target.__dict__)
        return (
            target,
            int(pixel_u),
            int(pixel_v),
            float(depth_m),
            float(base_point[0]),
            float(base_point[1]),
            float(base_point[2]),
        )

    def handle_click(self, depth_frame, intrinsics) -> Optional[ClickMeasurement]:
        target_pack = self._build_target_from_pixel(depth_frame, intrinsics)
        if target_pack is None:
            return None

        target, pixel_u, pixel_v, depth_m, base_x, base_y, base_z = target_pack

        step_result = None
        for _ in range(max(1, self.settle_steps)):
            step_result = self.runner.step(target)
            time.sleep(self.settle_sleep_s)
        if step_result is None:
            return None
        self.target = EndEffectorTarget(**step_result.safe_target.__dict__)

        current_qpos = self.runner.hardware.get_qpos_rad(self.runner.kin.arm.model.nq)
        actual_transform = self.runner.kin.fk(current_qpos)
        actual_x = float(actual_transform[0, 3])
        actual_y = float(actual_transform[1, 3])
        actual_z = float(actual_transform[2, 3])

        error_x = actual_x - step_result.safe_target.x
        error_y = actual_y - step_result.safe_target.y
        error_z = actual_z - step_result.safe_target.z
        error_norm = float(np.linalg.norm([error_x, error_y, error_z]))

        measurement = ClickMeasurement(
            index=len(self.measurements),
            pixel_u=int(pixel_u),
            pixel_v=int(pixel_v),
            depth_m=depth_m,
            predicted_base_x=float(step_result.safe_target.x),
            predicted_base_y=float(step_result.safe_target.y),
            predicted_base_z=float(step_result.safe_target.z),
            actual_base_x=actual_x,
            actual_base_y=actual_y,
            actual_base_z=actual_z,
            error_x_m=float(error_x),
            error_y_m=float(error_y),
            error_z_m=float(error_z),
            error_norm_m=error_norm,
            timestamp_s=float(time.time()),
        )
        self.measurements.append(measurement)
        self.last_error_mm = error_norm * 1000.0
        print(
            f"[#{measurement.index:03d}] "
            f"pred=({measurement.predicted_base_x:.4f},{measurement.predicted_base_y:.4f},{measurement.predicted_base_z:.4f}) "
            f"actual=({measurement.actual_base_x:.4f},{measurement.actual_base_y:.4f},{measurement.actual_base_z:.4f}) "
            f"err={measurement.error_norm_m*1000.0:.1f} mm"
        )
        return measurement

    def save_csv(self, output_path: Path) -> None:
        if not self.measurements:
            print("[WARN] No measurements to save.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(ClickMeasurement.__annotations__.keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.measurements:
                writer.writerow(row.__dict__)
        print(f"[INFO] Saved {len(self.measurements)} measurements -> {output_path}")

    def print_summary(self) -> None:
        if not self.measurements:
            print("[INFO] No measurements captured.")
            return
        errors_mm = np.array([measurement.error_norm_m * 1000.0 for measurement in self.measurements], dtype=float)
        print("========== Click Accuracy Summary ==========")
        print(f"samples: {len(errors_mm)}")
        print(f"mean   : {errors_mm.mean():.2f} mm")
        print(f"median : {np.median(errors_mm):.2f} mm")
        print(f"max    : {errors_mm.max():.2f} mm")
        print(f"p90    : {np.percentile(errors_mm, 90):.2f} mm")
        print("============================================")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate click-to-point accuracy with automatic error logging.")
    parser.add_argument("--output-csv", default="artifacts/calibration/vision_click_validation.csv")
    parser.add_argument("--settle-steps", type=int, default=20, help="Runner.step iterations after each click")
    parser.add_argument("--settle-sleep-s", type=float, default=0.02, help="Sleep duration between settle steps")
    parser.add_argument("--approach-height-m", type=float, default=0.10, help="Offset above clicked surface")
    parser.add_argument("--camera-window", default="Vision Click Validation")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    camera_to_base_transform = load_camera_to_base_transform()
    port = input(f"请输入 USB 端口 (回车默认 {DEFAULT_USB_PORT}): ").strip() or DEFAULT_USB_PORT

    kinematics = SO100Kinematics()
    hardware = SO100Hardware(port=port, use_degrees=True)
    hardware.connect()

    runner = CommonRunner(
        kin=kinematics,
        hardware=hardware,
        workspace=VISION_WORKSPACE,
        kp=DEFAULT_CONTROL_GAIN,
        filter_backend=DEFAULT_FILTER_BACKEND,
    )
    init_target = runner.initialize_from_robot()

    camera = D435Manager()
    validator = VisionClickValidator(
        runner=runner,
        init_target=init_target,
        camera_to_base_transform=camera_to_base_transform,
        settle_steps=arguments.settle_steps,
        settle_sleep_s=arguments.settle_sleep_s,
        approach_height_m=arguments.approach_height_m,
    )

    cv2.namedWindow(arguments.camera_window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(arguments.camera_window, validator.on_mouse)
    print("操作说明: 左键点击目标点进行测量；按 P 保存 CSV；按 Q/ESC 退出。")

    try:
        while True:
            depth_frame, color_frame = camera.get_frames()
            if not depth_frame or not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            if validator.clicked_pixel is not None:
                cv2.circle(image, validator.clicked_pixel, 6, (0, 0, 255), -1)
            cv2.putText(
                image,
                f"samples: {len(validator.measurements)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            if validator.last_error_mm is not None:
                cv2.putText(
                    image,
                    f"last err: {validator.last_error_mm:.1f} mm",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow(arguments.camera_window, image)

            measurement = validator.handle_click(depth_frame, camera.intrinsics)
            if measurement is not None:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("p"), ord("P")):
                validator.save_csv(Path(arguments.output_csv).expanduser().resolve())
                validator.print_summary()
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        validator.save_csv(Path(arguments.output_csv).expanduser().resolve())
        validator.print_summary()
        camera.stop()
        cv2.destroyAllWindows()
        hardware.disconnect()


if __name__ == "__main__":
    main()
