#!/usr/bin/env python3
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from common.common_control import CommonRunner
from common.config import (
    DEFAULT_CONTROL_GAIN,
    DEFAULT_FILTER_BACKEND,
    DEFAULT_USB_PORT,
    VISION_WORKSPACE,
    load_camera_to_base_transform,
)
from common.common_robot import EndEffectorTarget, SO100Hardware, SO100Kinematics


class D435Manager:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        return aligned.get_depth_frame(), aligned.get_color_frame()

    def stop(self):
        self.pipeline.stop()


class VisionTargetSource:
    def __init__(self, init_target: EndEffectorTarget, camera_to_base_transform: np.ndarray):
        self.target = EndEffectorTarget(**init_target.__dict__)
        self.clicked_pixel: Optional[Tuple[int, int]] = None
        self.camera_to_base_transform = camera_to_base_transform

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_pixel = (x, y)

    def update_from_click(self, depth_frame, intrinsics) -> Optional[EndEffectorTarget]:
        if self.clicked_pixel is None:
            return None
        u, v = self.clicked_pixel
        self.clicked_pixel = None
        depth = depth_frame.get_distance(u, v)
        if depth <= 0:
            print(f"\n[WARN] 点击像素({u}, {v})深度无效，请换点重试。")
            return None
        cam_pt = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
        p_cam = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
        p_base = self.camera_to_base_transform @ p_cam
        self.target.x = float(p_base[0])
        self.target.y = float(p_base[1])
        self.target.z = float(p_base[2] + 0.10)
        self.target.source = 'vision_click'
        self.target.timestamp = time.time()
        print(
            f"\n[INFO] 点击像素({u}, {v}) depth={depth:.3f}m -> "
            f"base_target=({self.target.x:.3f}, {self.target.y:.3f}, {self.target.z:.3f})"
        )
        return EndEffectorTarget(**self.target.__dict__)


def main():
    camera_to_base_transform = load_camera_to_base_transform()
    port = input(f'请输入 USB 端口 (回车默认 {DEFAULT_USB_PORT}): ').strip() or DEFAULT_USB_PORT
    kin = SO100Kinematics()
    hardware = SO100Hardware(port=port, use_degrees=True)
    hardware.connect()
    runner = CommonRunner(
        kin=kin,
        hardware=hardware,
        workspace=VISION_WORKSPACE,
        kp=DEFAULT_CONTROL_GAIN,
        filter_backend=DEFAULT_FILTER_BACKEND,
    )
    init_target = runner.initialize_from_robot()
    cam = D435Manager()
    source = VisionTargetSource(init_target, camera_to_base_transform)
    settle_steps = 20
    settle_sleep_s = 0.02

    cv2.namedWindow('Vision Target', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Vision Target', source.on_mouse)

    try:
        while True:
            depth_frame, color_frame = cam.get_frames()
            if not depth_frame or not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            maybe_target = source.update_from_click(depth_frame, cam.intrinsics)
            if maybe_target is not None:
                result = None
                for _ in range(settle_steps):
                    result = runner.step(maybe_target)
                    time.sleep(settle_sleep_s)
                if result is not None:
                    print(
                        f"[INFO] 收敛完成 -> SAFE "
                        f"[{result.safe_target.x:.3f}, {result.safe_target.y:.3f}, {result.safe_target.z:.3f}] "
                        f"ik_ok={result.ik_ok}"
                    )
            cv2.circle(img, (320, 240), 2, (0, 255, 0), -1)
            cv2.imshow('Vision Target', img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        hardware.disconnect()
        sys.exit(0)


if __name__ == '__main__':
    main()
