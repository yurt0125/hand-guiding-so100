"""Centralized runtime configuration for SO100 apps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

KEYBOARD_WORKSPACE: Dict[str, float] = {
    "x_min": -0.5,
    "x_max": 0.5,
    "y_min": -0.5,
    "y_max": 0.5,
    "z_min": 0.01,
    "z_max": 0.5,
}

VISION_WORKSPACE: Dict[str, float] = {
    "x_min": -0.4,
    "x_max": 0.4,
    "y_min": -0.4,
    "y_max": 0.0,
    "z_min": 0.01,
    "z_max": 0.4,
}

CSV_REPLAY_WORKSPACE: Dict[str, float] = {
    "x_min": -0.20,
    "x_max": 0.20,
    "y_min": -0.40,
    "y_max": -0.15,
    "z_min": 0.03,
    "z_max": 0.30,
}

DEFAULT_CONTROL_GAIN: float = 0.5
DEFAULT_USB_PORT: str = "/dev/ttyACM0"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_CAMERA_TO_BASE_TRANSFORM: np.ndarray = np.array(
    [
        [-0.9976, -0.0651, 0.0229, -0.0972],
        [-0.0662, 0.8080, -0.5854, -0.0089],
        [0.0196, -0.5856, -0.8104, 0.3828],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

DEFAULT_CAMERA_TO_BASE_PATHS: tuple[Path, ...] = (
    PROJECT_ROOT / "calibration" / "outputs" / "cam2base_latest.json",
    PROJECT_ROOT / "record" / "json" / "cam2base.json",
)

REPLAY_Y_MAX_THRESHOLD: float = -0.15
REPLAY_TARGET_OFFSET_DEFAULT_X: float = 0.0
REPLAY_TARGET_OFFSET_DEFAULT_Y: float = 0.0
REPLAY_TARGET_OFFSET_DEFAULT_Z: float = 0.0




DEFAULT_RECORD_FPS: int = 30
DEFAULT_RECORD_WIDTH: int = 640
DEFAULT_RECORD_HEIGHT: int = 480
DEFAULT_RECORD_EPISODE_TIME_S: float = 20.0
DEFAULT_RECORD_RESET_TIME_S: float = 10.0
DEFAULT_RECORD_NUM_EPISODES: int = 5
DEFAULT_RECORD_WARMUP_FRAMES: int = 30
DEFAULT_RECORD_DEPTH_MAX_M: float = 2.0
DEFAULT_RECORD_DEPTH_MIN_M: float = 0.1

DEFAULT_EXTRACT_FPS_FALLBACK: float = 30.0
DEFAULT_PATCH_RADIUS: int = 1
DEFAULT_VISUALIZE_EXPORT_FPS: float = 30.0
DEFAULT_FILTER_BACKEND: str = "guard_only"  # options: guard_only, kalman_guard
DEFAULT_KALMAN_PROCESS_NOISE: float = 1e-3
DEFAULT_KALMAN_MEASUREMENT_NOISE: float = 5e-3


@dataclass(frozen=True)
class RobotProfile:
    name: str
    arm_joint_names: tuple[str, ...]
    joint_max_speed_deg_s: tuple[float, ...]
    default_guess_q_rad: tuple[float, ...]


ROBOT_PROFILES: dict[str, RobotProfile] = {
    "so100_5dof": RobotProfile(
        name="so100_5dof",
        arm_joint_names=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ),
        joint_max_speed_deg_s=(80.0, 80.0, 100.0, 120.0, 180.0),
        default_guess_q_rad=(0.0547, -0.4193, 0.6309, -0.2114, -1.5357, 0.0),
    ),
    "so100_plus_6dof": RobotProfile(
        name="so100_plus_6dof",
        arm_joint_names=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "wrist_yaw",
        ),
        joint_max_speed_deg_s=(80.0, 80.0, 100.0, 120.0, 180.0, 180.0),
        default_guess_q_rad=(0.0547, -0.4193, 0.6309, -0.2114, -1.5357, 0.0, 0.0),
    ),
}

ACTIVE_ROBOT_PROFILE_NAME: str = "so100_5dof"


def get_active_robot_profile() -> RobotProfile:
    if ACTIVE_ROBOT_PROFILE_NAME not in ROBOT_PROFILES:
        return ROBOT_PROFILES["so100_5dof"]
    return ROBOT_PROFILES[ACTIVE_ROBOT_PROFILE_NAME]


def get_default_camera_to_base_path() -> Path:
    return DEFAULT_CAMERA_TO_BASE_PATHS[0]


def load_camera_to_base_transform() -> np.ndarray:
    for calibration_path in DEFAULT_CAMERA_TO_BASE_PATHS:
        if not calibration_path.exists():
            continue
        try:
            with calibration_path.open("r", encoding="utf-8") as calibration_file:
                payload = json.load(calibration_file)
            matrix = np.asarray(payload["T_cam2base"], dtype=float)
            if matrix.shape == (4, 4):
                return matrix
        except Exception:
            continue
    return DEFAULT_CAMERA_TO_BASE_TRANSFORM.copy()


