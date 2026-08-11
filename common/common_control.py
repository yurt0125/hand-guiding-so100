import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .common_robot import ACTIVE_ARM_JOINT_NAMES, ACTIVE_ROBOT_PROFILE, EndEffectorTarget, SO100Math
from .config import (
    DEFAULT_FILTER_BACKEND,
    DEFAULT_KALMAN_MEASUREMENT_NOISE,
    DEFAULT_KALMAN_PROCESS_NOISE,
)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def _normalize_axis(a: np.ndarray) -> np.ndarray:
    """Normalize a 3-vector; return default-down on zero input."""
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return a / norm


def _axis_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two unit vectors."""
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _slerp_axis(a: np.ndarray, b: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotate *a* toward *b* by at most *max_angle* radians along the great circle."""
    angle = _axis_angle_between(a, b)
    if angle < 1e-9:
        return a.copy()
    if angle <= max_angle:
        return b.copy()
    t = max_angle / angle
    # True spherical linear interpolation (SLERP)
    sin_angle = np.sin(angle)
    if sin_angle < 1e-9:
        return a.copy()
    result = (np.sin((1.0 - t) * angle) * a + np.sin(t * angle) * b) / sin_angle
    return _normalize_axis(result)


# ---------------------------------------------------------------------------
# PoseSmoother — EMA on position + vector EMA on tool axis
# ---------------------------------------------------------------------------

class PoseSmoother:
    def __init__(self, alpha_pos=0.25, alpha_ori=0.20, max_step_xyz=0.006, max_step_axis_deg=2.5):
        self.alpha_pos = float(alpha_pos)
        self.alpha_ori = float(alpha_ori)
        self.max_step_xyz = float(max_step_xyz)
        self.max_step_axis = math.radians(max_step_axis_deg)
        self.initialized = False
        self.xyz = np.zeros(3, dtype=float)
        self.axis_smooth = np.array([0.0, 0.0, -1.0])

    def reset(self, target: EndEffectorTarget) -> None:
        self.xyz[:] = [target.x, target.y, target.z]
        self.axis_smooth = target.tool_axis()
        self.initialized = True

    def update(self, target: EndEffectorTarget) -> EndEffectorTarget:
        if not self.initialized:
            self.reset(target)
            return target

        # Position EMA with rate limiting
        raw_xyz = np.array([target.x, target.y, target.z], dtype=float)
        ema_xyz = self.alpha_pos * raw_xyz + (1.0 - self.alpha_pos) * self.xyz
        dxyz = ema_xyz - self.xyz
        dnorm = np.linalg.norm(dxyz)
        if dnorm > self.max_step_xyz and dnorm > 1e-9:
            dxyz = dxyz / dnorm * self.max_step_xyz
        self.xyz += dxyz

        # Tool-axis vector EMA + renormalize + rate limiting
        a_raw = target.tool_axis()
        a_blended = self.alpha_ori * a_raw + (1.0 - self.alpha_ori) * self.axis_smooth
        a_blended = _normalize_axis(a_blended)
        self.axis_smooth = _slerp_axis(self.axis_smooth, a_blended, self.max_step_axis)

        return EndEffectorTarget(
            x=float(self.xyz[0]), y=float(self.xyz[1]), z=float(self.xyz[2]),
            tool_axis_x=float(self.axis_smooth[0]),
            tool_axis_y=float(self.axis_smooth[1]),
            tool_axis_z=float(self.axis_smooth[2]),
            gripper=float(target.gripper),
            valid=target.valid, source=target.source, timestamp=target.timestamp,
        )


# ---------------------------------------------------------------------------
# TrajectoryGuard — workspace clip + velocity limiting (position & axis)
# ---------------------------------------------------------------------------

class TrajectoryGuard:
    def __init__(self, workspace: Dict[str, float], max_xyz_speed=0.18, max_axis_speed_deg=100.0):
        self.workspace = workspace
        self.max_xyz_speed = float(max_xyz_speed)
        self.max_axis_speed = math.radians(max_axis_speed_deg)
        self.last_t = None
        self.last_xyz = None
        self.last_axis = None

    def reset(self, target: EndEffectorTarget) -> None:
        self.last_t = time.time()
        self.last_xyz = np.array([target.x, target.y, target.z], dtype=float)
        self.last_axis = target.tool_axis()

    def apply(self, target: EndEffectorTarget) -> EndEffectorTarget:
        # Workspace clipping
        xyz = np.array([target.x, target.y, target.z], dtype=float)
        xyz[0] = np.clip(xyz[0], self.workspace['x_min'], self.workspace['x_max'])
        xyz[1] = np.clip(xyz[1], self.workspace['y_min'], self.workspace['y_max'])
        xyz[2] = np.clip(xyz[2], self.workspace['z_min'], self.workspace['z_max'])

        a_target = target.tool_axis()

        if self.last_t is None:
            self.reset(target)
            return EndEffectorTarget(
                x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                tool_axis_x=float(a_target[0]),
                tool_axis_y=float(a_target[1]),
                tool_axis_z=float(a_target[2]),
                gripper=target.gripper, valid=target.valid,
                source=target.source, timestamp=target.timestamp,
            )

        now = time.time()
        dt = max(1e-3, now - self.last_t)

        # Position velocity limiting
        dxyz = xyz - self.last_xyz
        max_dist = self.max_xyz_speed * dt
        dnorm = np.linalg.norm(dxyz)
        if dnorm > max_dist and dnorm > 1e-9:
            xyz = self.last_xyz + dxyz / dnorm * max_dist

        # Axis angular velocity limiting (rotate along great circle)
        max_angle = self.max_axis_speed * dt
        a_safe = _slerp_axis(self.last_axis, a_target, max_angle)

        self.last_t = now
        self.last_xyz = xyz.copy()
        self.last_axis = a_safe.copy()

        return EndEffectorTarget(
            x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
            tool_axis_x=float(a_safe[0]),
            tool_axis_y=float(a_safe[1]),
            tool_axis_z=float(a_safe[2]),
            gripper=float(target.gripper), valid=target.valid,
            source=target.source, timestamp=target.timestamp,
        )


# ---------------------------------------------------------------------------
# KalmanPoseFilter — constant-velocity Kalman on position + axis
# ---------------------------------------------------------------------------

class KalmanPoseFilter:
    """
    Constant-velocity Kalman filter.
    state = [x, y, z, ax, ay, az, vx, vy, vz, vax, vay, vaz]  (12D)
    measurement = [x, y, z, ax, ay, az]                         (6D)

    After each update the axis portion (state[3:6]) is re-normalised
    onto the unit sphere.
    """

    N_STATE = 12
    N_MEAS = 6

    def __init__(self, process_noise: float = DEFAULT_KALMAN_PROCESS_NOISE,
                 measurement_noise: float = DEFAULT_KALMAN_MEASUREMENT_NOISE):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initialized = False
        self.last_t = None
        self.state = np.zeros(self.N_STATE, dtype=float)
        self.cov = np.eye(self.N_STATE, dtype=float) * 1e-3

    def reset(self, target: EndEffectorTarget) -> None:
        self.state[:] = 0.0
        self.state[0] = float(target.x)
        self.state[1] = float(target.y)
        self.state[2] = float(target.z)
        axis = target.tool_axis()
        self.state[3:6] = axis
        # velocities initialised to 0
        self.cov = np.eye(self.N_STATE, dtype=float) * 1e-3
        self.last_t = time.time()
        self.initialized = True

    def _build_transition(self, dt: float) -> np.ndarray:
        A = np.eye(self.N_STATE, dtype=float)
        for i in range(6):
            A[i, i + 6] = dt
        return A

    def update(self, target: EndEffectorTarget) -> EndEffectorTarget:
        if not self.initialized:
            self.reset(target)
            return target

        now = time.time()
        dt = max(1e-3, now - self.last_t) if self.last_t is not None else 1e-3
        self.last_t = now

        # Predict
        A = self._build_transition(dt)
        Q = np.eye(self.N_STATE, dtype=float) * self.process_noise
        self.state = A @ self.state
        self.cov = A @ self.cov @ A.T + Q

        # Measurement
        a_meas = target.tool_axis()
        measurement = np.array([
            target.x, target.y, target.z,
            a_meas[0], a_meas[1], a_meas[2],
        ], dtype=float)
        innovation = measurement - self.state[:6]

        H = np.zeros((self.N_MEAS, self.N_STATE), dtype=float)
        for i in range(self.N_MEAS):
            H[i, i] = 1.0
        R = np.eye(self.N_MEAS, dtype=float) * self.measurement_noise

        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.pinv(S)
        self.state = self.state + K @ innovation
        self.cov = (np.eye(self.N_STATE) - K @ H) @ self.cov

        # Re-normalise axis part onto unit sphere
        self.state[3:6] = _normalize_axis(self.state[3:6])

        return EndEffectorTarget(
            x=float(self.state[0]),
            y=float(self.state[1]),
            z=float(self.state[2]),
            tool_axis_x=float(self.state[3]),
            tool_axis_y=float(self.state[4]),
            tool_axis_z=float(self.state[5]),
            gripper=float(target.gripper),
            valid=target.valid,
            source=target.source,
            timestamp=target.timestamp,
        )


# ---------------------------------------------------------------------------
# FilterCoordinator
# ---------------------------------------------------------------------------

class FilterCoordinator:
    def __init__(self, workspace: Dict[str, float], backend: str = DEFAULT_FILTER_BACKEND):
        self.backend = str(backend)
        self.guard = TrajectoryGuard(workspace=workspace)
        self.kalman_filter = KalmanPoseFilter()

    def reset(self, target: EndEffectorTarget) -> None:
        self.guard.reset(target)
        self.kalman_filter.reset(target)

    def apply(self, target: EndEffectorTarget) -> EndEffectorTarget:
        if self.backend == "kalman_guard":
            filtered_target = self.kalman_filter.update(target)
            return self.guard.apply(filtered_target)
        return self.guard.apply(target)


# ---------------------------------------------------------------------------
# JointCommandFilter — per-joint rate limiting (unchanged)
# ---------------------------------------------------------------------------

class JointCommandFilter:
    def __init__(self, joint_names: List[str], max_speed_deg_s: List[float]):
        self.joint_names = list(joint_names)
        self.max_speed = {n: float(v) for n, v in zip(joint_names, max_speed_deg_s)}
        self.last_cmd = None
        self.last_t = None

    def reset(self, joint_deg: Dict[str, float]) -> None:
        self.last_cmd = {k: float(joint_deg[k]) for k in self.joint_names}
        self.last_t = time.time()

    def step(self, target_deg: Dict[str, float]) -> Dict[str, float]:
        now = time.time()
        if self.last_cmd is None:
            self.reset(target_deg)
            return dict(target_deg)
        dt = max(1e-3, now - self.last_t)
        out = {}
        for name in self.joint_names:
            prev = self.last_cmd[name]
            tgt = float(target_deg[name])
            max_delta = self.max_speed[name] * dt
            out[name] = prev + float(np.clip(tgt - prev, -max_delta, max_delta))
        self.last_cmd = out.copy()
        self.last_t = now
        return out


# ---------------------------------------------------------------------------
# ControlStepResult
# ---------------------------------------------------------------------------

@dataclass
class ControlStepResult:
    raw_target: EndEffectorTarget
    smooth_target: EndEffectorTarget
    safe_target: EndEffectorTarget
    dof_rad: np.ndarray
    joint_deg_cmd: Dict[str, float]
    ik_ok: bool
    mapping_quality: Dict[str, float]


# ---------------------------------------------------------------------------
# CommonRunner — main control orchestrator
# ---------------------------------------------------------------------------

class CommonRunner:
    def __init__(self, kin, hardware, workspace: Dict[str, float], kp: float = 0.5, filter_backend: str = DEFAULT_FILTER_BACKEND):
        self.kin = kin
        self.hardware = hardware
        self.kp = float(kp)
        self.pose_smoother = PoseSmoother()
        self.filter = FilterCoordinator(workspace=workspace, backend=filter_backend)
        self.joint_names = list(ACTIVE_ARM_JOINT_NAMES)
        self.joint_filter = JointCommandFilter(
            self.joint_names,
            list(ACTIVE_ROBOT_PROFILE.joint_max_speed_deg_s),
        )
        self.initialized = False
        self.last_dof = kin.last_dof.copy()
        self.workspace = workspace

    def initialize_from_robot(self):
        current_deg = self.hardware.get_joint_math_deg()
        q = self.hardware.get_qpos_rad(self.kin.arm.model.nq)
        tf = self.kin.fk(q)
        tool_axis = tf[:3, 2]  # z-column of rotation = tool approach direction
        target = EndEffectorTarget(
            x=float(tf[0, 3]), y=float(tf[1, 3]), z=float(tf[2, 3]),
            tool_axis_x=float(tool_axis[0]),
            tool_axis_y=float(tool_axis[1]),
            tool_axis_z=float(tool_axis[2]),
            gripper=float(current_deg['gripper']),
            valid=True, source='init',
        )
        self.pose_smoother.reset(target)
        self.filter.reset(target)
        self.joint_filter.reset({name: current_deg[name] for name in self.joint_names})
        self.last_dof = q.copy()
        self.initialized = True
        return target

    def safe_ik(self, p_target: np.ndarray, axis_target: np.ndarray,
                current_qpos_rad: np.ndarray, max_jump_rad: float = 0.8,
                max_pos_err: float = 0.01, max_axis_err_deg: float = 10.0):
        """IK with jump detection and post-solve verification.

        Returns (dof, success, jump_value).
        """
        dof, success = self.kin.ik(p_target, axis_target, self.last_dof)
        if not success:
            return current_qpos_rad.copy(), False, float("inf")

        # Joint jump check
        controlled_joint_count = len(self.joint_names)
        jump_value = float(
            np.max(np.abs(dof[:controlled_joint_count] - current_qpos_rad[:controlled_joint_count]))
        )
        if jump_value > max_jump_rad:
            return current_qpos_rad.copy(), False, jump_value

        # Post-solve verification: FK the solution and check actual errors
        tf_result = self.kin.fk(dof)
        pos_err = float(np.linalg.norm(tf_result[:3, 3] - p_target))
        axis_result = tf_result[:3, 2]
        axis_err_rad = float(np.arccos(np.clip(np.dot(axis_result, axis_target), -1.0, 1.0)))
        if pos_err > max_pos_err or axis_err_rad > np.radians(max_axis_err_deg):
            return current_qpos_rad.copy(), False, jump_value

        self.last_dof = dof.copy()
        return dof, True, jump_value

    def _compute_workspace_margin_score(self, target: EndEffectorTarget) -> float:
        x_span = max(1e-9, float(self.workspace["x_max"] - self.workspace["x_min"]))
        y_span = max(1e-9, float(self.workspace["y_max"] - self.workspace["y_min"]))
        z_span = max(1e-9, float(self.workspace["z_max"] - self.workspace["z_min"]))

        x_margin = min(target.x - self.workspace["x_min"], self.workspace["x_max"] - target.x) / x_span
        y_margin = min(target.y - self.workspace["y_min"], self.workspace["y_max"] - target.y) / y_span
        z_margin = min(target.z - self.workspace["z_min"], self.workspace["z_max"] - target.z) / z_span
        minimum_margin = min(x_margin, y_margin, z_margin)
        return float(np.clip(minimum_margin * 2.0, 0.0, 1.0))

    def _compute_mapping_quality(
        self,
        target: EndEffectorTarget,
        ik_ok: bool,
        jump_value_rad: float,
        max_jump_rad: float,
    ) -> Dict[str, float]:
        ik_score = 1.0 if ik_ok else 0.0
        jump_score = float(np.clip(1.0 - (jump_value_rad / max(max_jump_rad, 1e-9)), 0.0, 1.0))
        workspace_margin_score = self._compute_workspace_margin_score(target)
        total_score = 0.5 * ik_score + 0.3 * jump_score + 0.2 * workspace_margin_score
        singularity_risk = 1.0 - jump_score
        return {
            "score": float(np.clip(total_score, 0.0, 1.0)),
            "ik_score": float(ik_score),
            "jump_score": float(jump_score),
            "workspace_margin_score": float(workspace_margin_score),
            "singularity_risk_proxy": float(np.clip(singularity_risk, 0.0, 1.0)),
        }

    def step(self, raw_target: EndEffectorTarget) -> ControlStepResult:
        if not self.initialized:
            self.initialize_from_robot()

        current_deg = self.hardware.get_joint_math_deg()
        current_q = self.hardware.get_qpos_rad(self.kin.arm.model.nq)

        # Smooth → Filter → 5DOF IK (position + tool axis direction)
        smooth_target = self.pose_smoother.update(raw_target)
        safe_target = self.filter.apply(smooth_target)

        p_target = np.array([safe_target.x, safe_target.y, safe_target.z])
        axis_target = safe_target.tool_axis()
        max_jump_rad = 0.8
        dof, ik_ok, jump_value_rad = self.safe_ik(
            p_target, axis_target, current_q, max_jump_rad=max_jump_rad,
        )

        target_deg = {name: float(math.degrees(dof[i])) for i, name in enumerate(self.joint_names)}
        filtered_deg = self.joint_filter.step(target_deg)
        cmd_deg = {}
        for name in self.joint_names:
            curr = float(current_deg[name])
            cmd_deg[name] = curr + self.kp * (filtered_deg[name] - curr)

        self.hardware.send_math_deg_action(cmd_deg, safe_target.gripper)
        mapping_quality = self._compute_mapping_quality(
            target=safe_target,
            ik_ok=ik_ok,
            jump_value_rad=jump_value_rad,
            max_jump_rad=max_jump_rad,
        )
        return ControlStepResult(
            raw_target=raw_target,
            smooth_target=smooth_target,
            safe_target=safe_target,
            dof_rad=dof,
            joint_deg_cmd=cmd_deg,
            ik_ok=ik_ok,
            mapping_quality=mapping_quality,
        )
