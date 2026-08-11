#!/usr/bin/env python3
import os
import sys
import argparse
import time

# Keep project directory clean: no .pyc and no ultralytics config artifacts here.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_cfg")

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    raise SystemExit(f"[ERROR] pyrealsense2 import failed: {e}")

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(f"[ERROR] ultralytics import failed: {e}")

try:
    import open3d as o3d
except Exception:
    o3d = None


def parse_args():
    p = argparse.ArgumentParser("YOLO-seg + RealSense depth point cloud + ICP test")
    p.add_argument("--model", type=str, default="yolov8n-seg.pt", help="YOLO-seg model path")
    p.add_argument("--serial", type=str, default="", help="RealSense serial number (optional)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--target-class", type=str, default="banana", help="class name to keep")
    p.add_argument("--device", type=str, default="0", help="ultralytics device, e.g. 0 or cpu")
    p.add_argument("--mask-thr", type=float, default=0.5, help="mask binarization threshold")
    p.add_argument("--depth-min", type=float, default=0.15, help="min depth in meters")
    p.add_argument("--depth-max", type=float, default=1.2, help="max depth in meters")
    p.add_argument("--max-points", type=int, default=8000, help="max points before downsample")
    p.add_argument("--voxel", type=float, default=0.005, help="voxel size (m) for ICP")
    p.add_argument(
        "--template-ply",
        type=str,
        default="",
        help="object template point cloud for ICP; if empty, use first valid frame as template",
    )
    p.add_argument("--icp-max-corres", type=float, default=0.03, help="ICP max correspondence distance (m)")
    p.add_argument("--axis-len", type=float, default=0.02, help="TF axis length in meters")
    return p.parse_args()


def open_realsense(serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    return pipeline, profile


def mask_depth_to_points(mask: np.ndarray, depth_m: np.ndarray, intr):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    ys = ys[valid]
    xs = xs[valid]
    z = z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    x = (xs - intr.ppx) * z / intr.fx
    y = (ys - intr.ppy) * z / intr.fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def to_o3d_pcd(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def downsample_points(points: np.ndarray, max_points: int):
    if points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def mat_to_euler_xyz(R: np.ndarray):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float32)


def project_points(pts_3d: np.ndarray, intr):
    z = pts_3d[:, 2]
    valid = z > 1e-6
    uv = np.zeros((pts_3d.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (pts_3d[:, 0] * intr.fx / np.maximum(z, 1e-6)) + intr.ppx
    uv[:, 1] = (pts_3d[:, 1] * intr.fy / np.maximum(z, 1e-6)) + intr.ppy
    return uv, valid


def transform_points(T: np.ndarray, pts: np.ndarray):
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts.T).T + t


def make_bbox_corners(min_bound: np.ndarray, max_bound: np.ndarray):
    x0, y0, z0 = min_bound
    x1, y1, z1 = max_bound
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def draw_3d_box(image, T_obj_to_cam: np.ndarray, bbox_corners_obj: np.ndarray, intr, color=(255, 200, 0), thickness=2):
    pts_cam = transform_points(T_obj_to_cam, bbox_corners_obj)
    uv, valid = project_points(pts_cam, intr)
    if not np.all(valid):
        return
    uv = uv.astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        cv2.line(image, tuple(uv[a]), tuple(uv[b]), color, thickness)


def draw_tf_axes_from_center(image, center_cam: np.ndarray, R_obj_to_cam: np.ndarray, intr, axis_len=0.02, thickness=3):
    center_cam = center_cam.astype(np.float32).reshape(3)
    ex = center_cam + (R_obj_to_cam[:, 0].astype(np.float32) * axis_len)
    ey = center_cam + (R_obj_to_cam[:, 1].astype(np.float32) * axis_len)
    ez = center_cam + (R_obj_to_cam[:, 2].astype(np.float32) * axis_len)
    pts_cam = np.vstack([center_cam, ex, ey, ez])
    uv, valid = project_points(pts_cam, intr)
    if not np.all(valid):
        return
    uv = uv.astype(np.int32)
    o = tuple(uv[0])
    cv2.line(image, o, tuple(uv[1]), (0, 0, 255), thickness)   # X red
    cv2.line(image, o, tuple(uv[2]), (0, 255, 0), thickness)   # Y green
    cv2.line(image, o, tuple(uv[3]), (255, 0, 0), thickness)   # Z blue


def main():
    args = parse_args()

    print(f"[INFO] loading model: {args.model}")
    model = YOLO(args.model)
    if o3d is None:
        print("[WARN] open3d not found. ICP pose will be disabled. Install with: pip install open3d")

    print("[INFO] opening RealSense stream...")
    pipeline, profile = open_realsense(args.serial, args.width, args.height, args.fps)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(
        f"[INFO] intrinsics fx={color_intr.fx:.1f} fy={color_intr.fy:.1f} "
        f"cx={color_intr.ppx:.1f} cy={color_intr.ppy:.1f}, depth_scale={depth_scale}"
    )

    win = "YOLO-seg RealSense"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_last = time.time()
    fps_smooth = 0.0
    template_pcd = None
    template_ready = False
    T_prev = np.eye(4, dtype=np.float64)
    bbox_corners_obj = None
    obj_center_obj = None

    if args.template_ply and o3d is not None:
        template_pcd = o3d.io.read_point_cloud(args.template_ply)
        if len(template_pcd.points) == 0:
            raise SystemExit(f"[ERROR] empty template point cloud: {args.template_ply}")
        obj_center_obj = np.asarray(template_pcd.get_center(), dtype=np.float32)
        min_b = np.asarray(template_pcd.get_min_bound(), dtype=np.float32)
        max_b = np.asarray(template_pcd.get_max_bound(), dtype=np.float32)
        bbox_corners_obj = make_bbox_corners(min_b, max_b)
        template_pcd = template_pcd.voxel_down_sample(args.voxel)
        template_ready = True
        print(f"[INFO] loaded template: {args.template_ply}, points={len(template_pcd.points)}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_m = depth_raw * depth_scale
            overlay = frame.copy()
            pose_msg = "POSE: N/A"

            results = model.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                project="/tmp/ultralytics_runs",
                name="yolo_seg_realsense",
                exist_ok=True,
            )
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
                confs = r.boxes.conf.detach().cpu().numpy()
                xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(int)

                masks_np = None
                if r.masks is not None and r.masks.data is not None:
                    masks_np = r.masks.data.detach().cpu().numpy()  # [N,H,W]

                names = r.names
                for i, cid in enumerate(cls_ids):
                    cname = names.get(cid, str(cid))
                    if cname != args.target_class:
                        continue

                    x1, y1, x2, y2 = xyxy[i]
                    confv = confs[i]

                    # draw mask
                    if masks_np is not None and i < masks_np.shape[0]:
                        m = masks_np[i] > args.mask_thr
                        m = m & (depth_m > args.depth_min) & (depth_m < args.depth_max)

                        color = np.zeros_like(overlay)
                        color[:, :] = (0, 255, 0)
                        overlay[m] = cv2.addWeighted(overlay, 0.45, color, 0.55, 0)[m]

                        pts = mask_depth_to_points(m, depth_m, color_intr)
                        if pts.shape[0] < 50:
                            continue
                        pts = downsample_points(pts, args.max_points)
                        center = pts.mean(axis=0)
                        pose_msg = f"CENTER xyz(m): [{center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f}]"

                        # Estimate pose via ICP if open3d is available
                        if o3d is not None:
                            src_pcd = to_o3d_pcd(pts).voxel_down_sample(args.voxel)
                            if len(src_pcd.points) < 30:
                                continue

                            if not template_ready:
                                template_pcd = src_pcd
                                obj_center_obj = np.asarray(template_pcd.get_center(), dtype=np.float32)
                                min_b = np.asarray(template_pcd.get_min_bound(), dtype=np.float32)
                                max_b = np.asarray(template_pcd.get_max_bound(), dtype=np.float32)
                                bbox_corners_obj = make_bbox_corners(min_b, max_b)
                                template_ready = True
                                T_prev = np.eye(4, dtype=np.float64)
                                pose_msg = "POSE: template initialized from first valid mask frame"
                                continue

                            # template -> current
                            reg = o3d.pipelines.registration.registration_icp(
                                template_pcd,
                                src_pcd,
                                args.icp_max_corres,
                                T_prev,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
                            )
                            T = reg.transformation
                            T_prev = T
                            t = T[:3, 3]
                            eul = np.degrees(mat_to_euler_xyz(T[:3, :3]))
                            pose_msg = (
                                f"SE3 t=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]m "
                                f"rpy=[{eul[0]:+.1f},{eul[1]:+.1f},{eul[2]:+.1f}]deg "
                                f"fit={reg.fitness:.2f}"
                            )
                            # Draw 6D oriented box + TF-like axes at object center.
                            if bbox_corners_obj is not None:
                                draw_3d_box(overlay, T, bbox_corners_obj, color_intr, color=(0, 255, 255), thickness=2)
                                draw_tf_axes_from_center(
                                    overlay,
                                    center_cam=center,
                                    R_obj_to_cam=T[:3, :3],
                                    intr=color_intr,
                                    axis_len=args.axis_len,
                                    thickness=3,
                                )

            # FPS
            now = time.time()
            dt = max(1e-6, now - t_last)
            inst = 1.0 / dt
            fps_smooth = inst if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst
            t_last = now

            cv2.putText(
                overlay,
                f"FPS: {fps_smooth:.1f} | class={args.target_class}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                pose_msg[:120],
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
            )

            cv2.imshow(win, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
