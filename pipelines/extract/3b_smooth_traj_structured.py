#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_default_camera_to_base_path


EPSILON = 1e-9


def load_transform_json(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matrix = np.asarray(payload["T_cam2base"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"T_cam2base must be shape (4, 4), got {matrix.shape}")
    return matrix


def read_intrinsics_from_episode(episode_dir: str) -> dict[str, float]:
    intrinsics_path = Path(episode_dir) / "intrinsics.json"
    with intrinsics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    intrinsics = payload["intrinsics"]
    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "ppx": float(intrinsics["ppx"]),
        "ppy": float(intrinsics["ppy"]),
    }


def project_point(point_xyz: np.ndarray, intrinsics: dict[str, float]) -> np.ndarray:
    x_value, y_value, z_value = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    safe_z = z_value if abs(z_value) > EPSILON else (EPSILON if z_value >= 0 else -EPSILON)
    u_value = intrinsics["fx"] * x_value / safe_z + intrinsics["ppx"]
    v_value = intrinsics["fy"] * y_value / safe_z + intrinsics["ppy"]
    return np.asarray([u_value, v_value], dtype=float)


def optimize_points(
    observed_point_series: np.ndarray,
    observed_uv_series: np.ndarray,
    temporal_smooth_weight: float,
    reprojection_weight: float,
    data_weight: float,
    intrinsics: dict[str, float],
    learning_rate: float,
    iteration_count: int,
    gradient_clip_value: float,
) -> np.ndarray:
    optimized_point_series = observed_point_series.copy()
    frame_count = optimized_point_series.shape[0]
    if frame_count <= 1:
        return optimized_point_series

    for _ in range(iteration_count):
        gradient_series = np.zeros_like(optimized_point_series)

        gradient_series += 2.0 * data_weight * (optimized_point_series - observed_point_series)

        temporal_deltas = optimized_point_series[1:] - optimized_point_series[:-1]
        gradient_series[1:] += 2.0 * temporal_smooth_weight * temporal_deltas
        gradient_series[:-1] -= 2.0 * temporal_smooth_weight * temporal_deltas

        for frame_index in range(frame_count):
            observed_uv = observed_uv_series[frame_index]
            if not np.isfinite(observed_uv).all():
                continue

            point_xyz = optimized_point_series[frame_index]
            x_value, y_value, z_value = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
            safe_z = z_value if abs(z_value) > EPSILON else (EPSILON if z_value >= 0 else -EPSILON)
            safe_z_square = safe_z * safe_z

            predicted_uv = project_point(point_xyz, intrinsics)
            uv_error = predicted_uv - observed_uv

            du_dx = intrinsics["fx"] / safe_z
            du_dz = -intrinsics["fx"] * x_value / safe_z_square
            dv_dy = intrinsics["fy"] / safe_z
            dv_dz = -intrinsics["fy"] * y_value / safe_z_square

            gradient_series[frame_index, 0] += 2.0 * reprojection_weight * uv_error[0] * du_dx
            gradient_series[frame_index, 1] += 2.0 * reprojection_weight * uv_error[1] * dv_dy
            gradient_series[frame_index, 2] += 2.0 * reprojection_weight * (
                uv_error[0] * du_dz + uv_error[1] * dv_dz
            )

        np.clip(gradient_series, -gradient_clip_value, gradient_clip_value, out=gradient_series)
        optimized_point_series -= learning_rate * gradient_series

    return optimized_point_series


def apply_bone_length_consistency(
    point_a_series: np.ndarray,
    point_b_series: np.ndarray,
    bone_weight: float,
    iteration_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    optimized_point_a = point_a_series.copy()
    optimized_point_b = point_b_series.copy()

    observed_bone_length = np.linalg.norm(point_a_series - point_b_series, axis=1)
    valid_length_mask = np.isfinite(observed_bone_length) & (observed_bone_length > 1e-6)
    if not np.any(valid_length_mask):
        return optimized_point_a, optimized_point_b
    target_bone_length = float(np.median(observed_bone_length[valid_length_mask]))

    for _ in range(iteration_count):
        bone_vector = optimized_point_a - optimized_point_b
        bone_length = np.linalg.norm(bone_vector, axis=1) + EPSILON
        bone_error = bone_length - target_bone_length

        correction_scale = bone_weight * bone_error / bone_length
        correction_vector = bone_vector * correction_scale[:, None]
        optimized_point_a -= correction_vector
        optimized_point_b += correction_vector

    return optimized_point_a, optimized_point_b


def run_structured_smoothing(arguments: argparse.Namespace) -> None:
    dataframe = pd.read_csv(arguments.input_csv)
    required_columns = [
        "hand_found",
        "kp02_cam_x",
        "kp02_cam_y",
        "kp02_cam_z",
        "kp05_cam_x",
        "kp05_cam_y",
        "kp05_cam_z",
        "kp02_u",
        "kp02_v",
        "kp05_u",
        "kp05_v",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    valid_row_mask = dataframe["hand_found"].astype(int).to_numpy() == 1
    if int(np.sum(valid_row_mask)) < 2:
        raise ValueError("Not enough valid rows (hand_found==1) to smooth.")

    valid_dataframe = dataframe.loc[valid_row_mask].copy()
    valid_index_values = valid_dataframe.index.to_numpy()

    kp02_observed_points = valid_dataframe[["kp02_cam_x", "kp02_cam_y", "kp02_cam_z"]].to_numpy(dtype=float)
    kp05_observed_points = valid_dataframe[["kp05_cam_x", "kp05_cam_y", "kp05_cam_z"]].to_numpy(dtype=float)
    kp02_observed_uv = valid_dataframe[["kp02_u", "kp02_v"]].to_numpy(dtype=float)
    kp05_observed_uv = valid_dataframe[["kp05_u", "kp05_v"]].to_numpy(dtype=float)

    intrinsics = read_intrinsics_from_episode(arguments.episode_dir)

    kp02_optimized_points = optimize_points(
        observed_point_series=kp02_observed_points,
        observed_uv_series=kp02_observed_uv,
        temporal_smooth_weight=arguments.lambda_smooth,
        reprojection_weight=arguments.lambda_2d,
        data_weight=arguments.lambda_data,
        intrinsics=intrinsics,
        learning_rate=arguments.learning_rate,
        iteration_count=arguments.iterations,
        gradient_clip_value=arguments.gradient_clip,
    )
    kp05_optimized_points = optimize_points(
        observed_point_series=kp05_observed_points,
        observed_uv_series=kp05_observed_uv,
        temporal_smooth_weight=arguments.lambda_smooth,
        reprojection_weight=arguments.lambda_2d,
        data_weight=arguments.lambda_data,
        intrinsics=intrinsics,
        learning_rate=arguments.learning_rate,
        iteration_count=arguments.iterations,
        gradient_clip_value=arguments.gradient_clip,
    )

    kp02_optimized_points, kp05_optimized_points = apply_bone_length_consistency(
        point_a_series=kp02_optimized_points,
        point_b_series=kp05_optimized_points,
        bone_weight=arguments.lambda_bone,
        iteration_count=arguments.bone_iterations,
    )

    midpoint_camera_points = 0.5 * (kp02_optimized_points + kp05_optimized_points)
    camera_to_base_transform = load_transform_json(arguments.transform_json)
    rotation_camera_to_base = camera_to_base_transform[:3, :3]
    translation_camera_to_base = camera_to_base_transform[:3, 3]
    midpoint_base_points = (rotation_camera_to_base @ midpoint_camera_points.T).T + translation_camera_to_base

    dataframe.loc[valid_index_values, ["kp02_cam_x", "kp02_cam_y", "kp02_cam_z"]] = kp02_optimized_points
    dataframe.loc[valid_index_values, ["kp05_cam_x", "kp05_cam_y", "kp05_cam_z"]] = kp05_optimized_points
    dataframe.loc[valid_index_values, ["mid_cam_x", "mid_cam_y", "mid_cam_z"]] = midpoint_camera_points
    dataframe.loc[valid_index_values, ["mid_base_x", "mid_base_y", "mid_base_z"]] = midpoint_base_points

    output_path = Path(arguments.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    original_midpoint = 0.5 * (kp02_observed_points + kp05_observed_points)
    displacement = np.linalg.norm(midpoint_camera_points - original_midpoint, axis=1)
    print(f"[INFO] Structured smoothing completed: {arguments.output_csv}")
    print(f"[INFO] Valid smoothed rows: {len(valid_index_values)}")
    print(f"[INFO] Mean midpoint displacement (m): {float(np.mean(displacement)):.6f}")
    print(f"[INFO] Max midpoint displacement (m): {float(np.max(displacement)):.6f}")


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Structured smoothing with temporal, bone-length and reprojection constraints.",
    )
    argument_parser.add_argument("--input-csv", required=True)
    argument_parser.add_argument("--episode-dir", required=True, help="must contain intrinsics.json")
    argument_parser.add_argument(
        "--transform-json",
        default=str(get_default_camera_to_base_path()),
        help="contains T_cam2base; default uses calibration/outputs/cam2base_latest.json",
    )
    argument_parser.add_argument("--output-csv", required=True)
    argument_parser.add_argument("--lambda-data", type=float, default=1.0)
    argument_parser.add_argument("--lambda-smooth", type=float, default=8.0)
    argument_parser.add_argument("--lambda-bone", type=float, default=0.15)
    argument_parser.add_argument("--lambda-2d", type=float, default=1e-6)
    argument_parser.add_argument("--learning-rate", type=float, default=2e-3)
    argument_parser.add_argument("--iterations", type=int, default=120)
    argument_parser.add_argument("--bone-iterations", type=int, default=20)
    argument_parser.add_argument("--gradient-clip", type=float, default=5.0)
    return argument_parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_structured_smoothing(arguments)


if __name__ == "__main__":
    main()
