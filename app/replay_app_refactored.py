#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
import threading
import math
from collections import deque
from typing import Optional
import shutil

# Avoid numerical backend oversubscription while replay, IK, UI, and optional
# RealSense recording run at the same time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import cv2
import mujoco
import numpy as np
import pandas as pd

from common.src import pinocchio_kinematic
from common.config import (
    VISION_WORKSPACE,
    REPLAY_TARGET_OFFSET_DEFAULT_X,
    REPLAY_TARGET_OFFSET_DEFAULT_Y,
    REPLAY_TARGET_OFFSET_DEFAULT_Z,
)
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML_PATH = str(PROJECT_ROOT / "common" / "model" / "trs_so_arm100" / "scene.xml")
ARM_XML_PATH = str(PROJECT_ROOT / "common" / "model" / "trs_so_arm100" / "so_arm100.xml")

DEFAULT_TOOL_AXIS = np.array([0.0, 0.0, -1.0])


def _normalize_axis(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return DEFAULT_TOOL_AXIS.copy()
    return a / norm


def _slerp_axis(a: np.ndarray, b: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotate *a* toward *b* by at most *max_angle* radians along the great circle."""
    angle = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    if angle < 1e-9:
        return a.copy()
    if angle <= max_angle:
        return b.copy()
    t = max_angle / angle
    sin_angle = np.sin(angle)
    if sin_angle < 1e-9:
        return a.copy()
    result = (np.sin((1.0 - t) * angle) * a + np.sin(t * angle) * b) / sin_angle
    return _normalize_axis(result)


def _lerp_axis(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend + renormalize — suitable for small-angle interpolation."""
    blended = (1.0 - alpha) * a + alpha * b
    return _normalize_axis(blended)


class ReplayVideoRecorder:
    """Optional RealSense recorder: capture one frame per replay action."""

    def __init__(self, output_path: str, serial: Optional[str], width: int, height: int, fps: float):
        self.output_path = Path(output_path).expanduser().resolve()
        self.map_csv_path = self.output_path.with_suffix(".map.csv")
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._pipeline = None
        self._writer = None
        self._active_episode: Optional[int] = None
        self._records: list[dict] = []
        self._segment_dir = self.output_path.with_suffix(".segments")
        self._episode_frame_counts: dict[int, int] = {}
        self.enabled = False

    def _segment_path(self, episode_index: int) -> Path:
        return self._segment_dir / f"episode_{int(episode_index):06d}.mp4"

    def _close_active_writer(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None
        self._active_episode = None

    def _open_writer_for_episode(self, episode_index: int):
        self._segment_dir.mkdir(parents=True, exist_ok=True)
        segment_path = self._segment_path(episode_index)
        if segment_path.exists():
            segment_path.unlink()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(segment_path), fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open segment writer: {segment_path}")
        self._writer = writer
        self._active_episode = int(episode_index)
        self._episode_frame_counts[int(episode_index)] = 0

    def start(self, resume: bool = False):
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            print(f"[WARN] replay video record disabled: pyrealsense2 import failed: {exc}")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, int(round(self.fps)))

        try:
            pipeline.start(config)
        except Exception as exc:
            print(f"[WARN] replay video record disabled: failed to start RealSense: {exc}")
            return

        self._pipeline = pipeline
        self._writer = None
        self._active_episode = None
        self._records.clear()
        self._episode_frame_counts.clear()
        if not resume:
            if self.output_path.exists():
                self.output_path.unlink()
            if self.map_csv_path.exists():
                self.map_csv_path.unlink()
            if self._segment_dir.exists():
                shutil.rmtree(self._segment_dir, ignore_errors=True)
        self.enabled = True
        print(
            "[INFO] replay video recording enabled: "
            f"{self.output_path} ({self.width}x{self.height}@{self.fps:.1f})"
        )

    def load_resume_records(self, records: list[dict]) -> None:
        self._records = []
        self._episode_frame_counts.clear()
        for r in records:
            rec = {
                "video_frame_index": int(r.get("video_frame_index", -1)),
                "episode_local_frame_index": int(r.get("episode_local_frame_index", -1)),
                "video_timestamp_s": float(r.get("video_timestamp_s", np.nan)),
                "action_seq": int(r.get("action_seq", -1)),
                "csv_row_index": int(r.get("csv_row_index", -1)),
                "abs_index": int(r.get("abs_index", -1)),
                "episode_index": int(r.get("episode_index", -1)),
            }
            self._records.append(rec)
            epi = rec["episode_index"]
            self._episode_frame_counts[epi] = self._episode_frame_counts.get(epi, 0) + 1

    def capture_one(self, action_seq: int, csv_row_index: int, abs_index: int, episode_index: int) -> tuple[int, float]:
        if not self.enabled or self._pipeline is None:
            return -1, float("nan")
        try:
            episode_index = int(episode_index)
            if self._active_episode != episode_index or self._writer is None:
                self._close_active_writer()
                self._open_writer_for_episode(episode_index)

            frames = self._pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return -1, float("nan")
            frame = np.asanyarray(color_frame.get_data())
            self._writer.write(frame)
            local_idx = int(self._episode_frame_counts.get(episode_index, 0))
            self._episode_frame_counts[episode_index] = local_idx + 1
            ts_s = float(color_frame.get_timestamp()) / 1000.0
            self._records.append(
                {
                    "video_frame_index": -1,
                    "episode_local_frame_index": local_idx,
                    "video_timestamp_s": ts_s,
                    "action_seq": int(action_seq),
                    "csv_row_index": int(csv_row_index),
                    "abs_index": int(abs_index),
                    "episode_index": episode_index,
                }
            )
            return -1, ts_s
        except Exception:
            return -1, float("nan")

    def drop_episode_records(self, episode_index: int) -> None:
        episode_index = int(episode_index)
        before = len(self._records)
        self._records = [r for r in self._records if int(r.get("episode_index", -1)) != episode_index]
        removed = before - len(self._records)
        if removed > 0:
            print(f"[record] dropped {removed} video-map rows for episode {episode_index}")
        seg = self._segment_path(episode_index)
        if seg.exists():
            seg.unlink()
            print(f"[record] removed video segment for episode {episode_index}: {seg}")
        self._episode_frame_counts.pop(episode_index, None)
        if self._active_episode == episode_index:
            self._close_active_writer()

    def _compose_final_video_and_map(self):
        if not self._records:
            return
        ordered_records = sorted(self._records, key=lambda r: int(r.get("action_seq", -1)))
        for i, rec in enumerate(ordered_records):
            rec["video_frame_index"] = int(i)
        pd.DataFrame.from_records(ordered_records).to_csv(self.map_csv_path, index=False)
        print(f"[INFO] replay video map saved: {self.map_csv_path} (rows={len(ordered_records)})")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        final_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        if not final_writer.isOpened():
            print(f"[WARN] failed to open final replay video writer: {self.output_path}")
            return

        episode_order: list[int] = []
        seen = set()
        for rec in ordered_records:
            e = int(rec.get("episode_index", -1))
            if e not in seen:
                seen.add(e)
                episode_order.append(e)

        written = 0
        for e in episode_order:
            seg = self._segment_path(e)
            if not seg.exists():
                print(f"[WARN] missing segment for episode {e}: {seg}")
                continue
            cap = cv2.VideoCapture(str(seg))
            if not cap.isOpened():
                print(f"[WARN] failed to open segment for episode {e}: {seg}")
                continue
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                final_writer.write(frame)
                written += 1
            cap.release()

        final_writer.release()
        print(f"[INFO] replay video saved: {self.output_path} (frames={written})")

    def stop(self):
        self._close_active_writer()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        if self.enabled:
            self._compose_final_video_and_map()


class ReplayController:
    def __init__(self):
        self.x, self.y, self.z = -0.021, -0.138, 0.102
        self.target_x, self.target_y, self.target_z = self.x, self.y, self.z

        self.tool_axis = DEFAULT_TOOL_AXIS.copy()
        self.target_tool_axis = DEFAULT_TOOL_AXIS.copy()

        self.gripper_pos = 0.0
        # Unified runtime semantic:
        #   internal [0, 1], send to LeRobot gripper.pos [0, 100] at hardware boundary.
        self.gripper_min, self.gripper_max = 0.0, 1.0

        # Align replay workspace with vision/keyboard operational workspace.
        self.x_min, self.x_max = float(VISION_WORKSPACE["x_min"]), float(VISION_WORKSPACE["x_max"])
        self.y_min, self.y_max = float(VISION_WORKSPACE["y_min"]), float(VISION_WORKSPACE["y_max"])
        self.z_min, self.z_max = float(VISION_WORKSPACE["z_min"]), float(VISION_WORKSPACE["z_max"])

    def clamp_pose(self, x_value, y_value, z_value):
        x_value = float(np.clip(x_value, self.x_min, self.x_max))
        y_value = float(np.clip(y_value, self.y_min, self.y_max))
        z_value = float(np.clip(z_value, self.z_min, self.z_max))
        return x_value, y_value, z_value

    def apply_pose(self, x_value, y_value, z_value, ax, ay, az, csv_gripper_cmd=None, immediate: bool = False):
        """Apply a new target from CSV (tool axis is absolute direction in base frame)."""
        x_value, y_value, z_value = self.clamp_pose(x_value, y_value, z_value)

        self.target_x, self.target_y, self.target_z = x_value, y_value, z_value
        axis = _normalize_axis(np.array([ax, ay, az]))
        self.target_tool_axis = axis

        if immediate:
            self.x, self.y, self.z = x_value, y_value, z_value
            self.tool_axis = axis.copy()

        if csv_gripper_cmd is not None and np.isfinite(csv_gripper_cmd):
            self.gripper_pos = float(np.clip(csv_gripper_cmd, self.gripper_min, self.gripper_max))


class TrajectoryReplayApp:
    def __init__(
        self,
        csv_path,
        real_robot=None,
        fps=5.0,
        target_offset_x: float = -0.0203,
        target_offset_y: float = 0.0,
        target_offset_z: float = 0.0,
        export_action_csv: Optional[str] = None,
        record_video_path: Optional[str] = None,
        record_rs_serial: Optional[str] = None,
        record_video_width: int = 640,
        record_video_height: int = 480,
        record_video_fps: float = 30.0,
        episode_reset_time_s: float = 0.0,
        resume: bool = False,
        profile_control: bool = False,
        auto_resync_on_joint_jump: bool = True,
    ):
        self.real_robot = real_robot
        self.controller = ReplayController()
        self.target_offset_x = float(target_offset_x)
        self.target_offset_y = float(target_offset_y)
        self.target_offset_z = float(target_offset_z)
        self.export_action_csv = Path(export_action_csv).expanduser().resolve() if export_action_csv else None
        self.sent_actions_records: list[dict] = []
        self.episode_reset_time_s = max(0.0, float(episode_reset_time_s))
        self.sent_action_seq = 0
        self.last_emitted_abs_index = -1
        self.resume = bool(resume)
        self.profile_control = bool(profile_control)
        self.auto_resync_on_joint_jump = bool(auto_resync_on_joint_jump)
        self.video_recorder: Optional[ReplayVideoRecorder] = None
        if record_video_path:
            self.video_recorder = ReplayVideoRecorder(
                output_path=record_video_path,
                serial=record_rs_serial,
                width=record_video_width,
                height=record_video_height,
                fps=record_video_fps,
            )
            self.video_recorder.start(resume=self.resume)

        self.is_running = True
        self.is_paused = True
        self.auto_mode = False
        self.step_once = False
        self.fps = fps

        self.status_lock = threading.Lock()
        self.current_status = {
            "row_idx": 0,
            "total_rows": 0,
            "x": self.controller.x,
            "y": self.controller.y,
            "z": self.controller.z,
            "axis": self.controller.tool_axis.tolist(),
            "state": "INIT",
            "mode": "STEP",
            "last_msg": "",
            "last_csv_idx": -1,
        }

        print("正在初始化后台 IK 逆解算器...")
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.arm = pinocchio_kinematic.Kinematics("JawOffset")
        self.arm.buildFromMJCF(ARM_XML_PATH)

        self.guess_q = np.array([0.0474, -3.0230, 2.8877, 0.7562, -1.5186, -0.0], dtype=float)
        self.last_dof = np.zeros(self.arm.model.nq)
        copy_len = min(len(self.guess_q), self.arm.model.nq)
        self.last_dof[:copy_len] = self.guess_q[:copy_len]
        self.data.qpos[:copy_len] = self.guess_q[:copy_len]
        mujoco.mj_forward(self.model, self.data)
        initial_transform = self.arm.fk(self.last_dof)
        self.controller.x = float(initial_transform[0, 3])
        self.controller.y = float(initial_transform[1, 3])
        self.controller.z = float(initial_transform[2, 3])
        self.controller.target_x = self.controller.x
        self.controller.target_y = self.controller.y
        self.controller.target_z = self.controller.z
        initial_axis = _normalize_axis(initial_transform[:3, 2])
        self.controller.tool_axis = initial_axis.copy()
        self.controller.target_tool_axis = initial_axis.copy()

        print(f"正在读取轨迹 CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self._ensure_tool_axis_columns()
        # Preserve original absolute row identity from abs.csv across all filtering/interpolation.
        self.df["abs_index"] = np.arange(len(self.df), dtype=np.int64)
        self._normalize_gripper_cmd_column_inplace()
        print(
            "[INFO] replay target offset (m): "
            f"dx={self.target_offset_x:+.4f}, dy={self.target_offset_y:+.4f}, dz={self.target_offset_z:+.4f}"
        )

        required_cols = ["mid_base_x", "mid_base_y", "mid_base_z", "tool_axis_x", "tool_axis_y", "tool_axis_z"]
        for column_name in required_cols:
            if column_name not in self.df.columns:
                raise ValueError(f"CSV 缺少列: {column_name}")

        self.valid_rows = self.df.copy()
        if "hand_found" in self.valid_rows.columns:
            self.valid_rows = self.valid_rows[self.valid_rows["hand_found"] == 1].reset_index(drop=True)

        if len(self.valid_rows) == 0:
            raise ValueError("CSV 中没有可用轨迹行")

        self.row_idx = 0
        self.last_frame_time = time.time()
        self.last_control_time = time.time()
        self.last_send_time = 0.0
        self.has_applied_first_target = False

        # Smoothing parameters (do not alter main control backbone)
        self.enable_soft_y_filter = False
        self.soft_y_limit = -0.15
        self.bootstrap_transition_steps = 0
        self.max_position_step_m = 0.008
        self.max_axis_step_rad = math.radians(2.0)
        self.max_candidate_joint_jump_rad = 0.9
        self.bootstrap_queue = []
        self.interpolated_target_queue = deque()
        self.previous_valid_target = None
        self.resync_target = None
        self.current_episode_idx: Optional[int] = None
        self._skip_reset_event = threading.Event()
        self._is_reset_waiting = False
        self._redo_prev_episode_event = threading.Event()
        self.recording_enabled = bool(self.export_action_csv is not None or self.video_recorder is not None)

        with self.status_lock:
            self.current_status["total_rows"] = len(self.valid_rows)

        self.episode_row_ranges = self._build_episode_row_ranges()
        self._load_resume_state_if_needed()

        self.robot_thread = threading.Thread(target=self.robot_control_loop, daemon=True)
        self.robot_thread.start()

    def _load_resume_state_if_needed(self):
        if not self.resume:
            return
        if self.export_action_csv is None or not self.export_action_csv.exists():
            print("[WARN] --resume enabled but export-action-csv missing; start from beginning.")
            return
        try:
            df_prev = pd.read_csv(self.export_action_csv)
        except Exception as exc:
            print(f"[WARN] failed to load previous sent_action csv: {exc}; start from beginning.")
            return
        if df_prev.empty or "episode_index" not in df_prev.columns:
            print("[WARN] previous sent_action csv empty or no episode_index; start from beginning.")
            return

        df_prev = df_prev.sort_values("action_seq").reset_index(drop=True) if "action_seq" in df_prev.columns else df_prev.reset_index(drop=True)
        epi_series = pd.to_numeric(df_prev["episode_index"], errors="coerce")
        if not np.isfinite(epi_series.to_numpy(dtype=float)).any():
            print("[WARN] previous sent_action episode_index invalid; start from beginning.")
            return
        last_episode = int(np.nanmax(epi_series.to_numpy(dtype=float)))
        df_keep = df_prev[epi_series != last_episode].copy().reset_index(drop=True)

        # Renumber action_seq to keep it continuous.
        if len(df_keep) > 0:
            df_keep["action_seq"] = np.arange(len(df_keep), dtype=np.int64)
        self.sent_actions_records = df_keep.to_dict(orient="records")
        self.sent_action_seq = len(self.sent_actions_records)

        # Resume replay pointer from the start of the last (possibly partial) episode.
        resume_episode = last_episode
        if resume_episode in self.episode_row_ranges:
            self.row_idx = int(self.episode_row_ranges[resume_episode][0])
            self.current_episode_idx = resume_episode
            self.update_status("RESUME", f"resume from episode {resume_episode}, kept rows={len(df_keep)}")
            print(f"[INFO] resume: rewind to episode {resume_episode}, kept previous action rows={len(df_keep)}")
        else:
            print(f"[WARN] resume episode {resume_episode} not found in current csv; start from beginning.")
            self.row_idx = 0
            self.current_episode_idx = None
            self.sent_actions_records = []
            self.sent_action_seq = 0
            return

        if self.video_recorder is not None:
            # Prefer previous map.csv if available; fallback to kept sent_action rows.
            map_path = self.video_recorder.map_csv_path
            loaded_records: list[dict] = []
            if map_path.exists():
                try:
                    map_df = pd.read_csv(map_path)
                    if len(map_df) > 0 and "episode_index" in map_df.columns:
                        map_epi = pd.to_numeric(map_df["episode_index"], errors="coerce")
                        map_df = map_df[map_epi != last_episode].copy().reset_index(drop=True)
                        if "action_seq" not in map_df.columns:
                            map_df["action_seq"] = np.arange(len(map_df), dtype=np.int64)
                        else:
                            map_df["action_seq"] = np.arange(len(map_df), dtype=np.int64)
                        loaded_records = map_df.to_dict(orient="records")
                except Exception as exc:
                    print(f"[WARN] failed to load map for resume: {exc}")
            if not loaded_records and len(df_keep) > 0:
                temp = df_keep.copy()
                if "video_timestamp_s" not in temp.columns:
                    temp["video_timestamp_s"] = np.nan
                if "video_frame_index" not in temp.columns:
                    temp["video_frame_index"] = -1
                if "episode_local_frame_index" not in temp.columns:
                    temp["episode_local_frame_index"] = -1
                for c in ["csv_row_index", "abs_index"]:
                    if c not in temp.columns:
                        temp[c] = -1
                loaded_records = temp[[
                    "video_frame_index",
                    "episode_local_frame_index",
                    "video_timestamp_s",
                    "action_seq",
                    "csv_row_index",
                    "abs_index",
                    "episode_index",
                ]].to_dict(orient="records")
            if loaded_records:
                self.video_recorder.load_resume_records(loaded_records)

    def _resolve_current_episode_for_redo(self) -> Optional[int]:
        if self.current_episode_idx is not None:
            return int(self.current_episode_idx)
        if len(self.valid_rows) == 0 or "episode_index" not in self.valid_rows.columns:
            return None
        probe_idx = int(np.clip(self.row_idx, 0, len(self.valid_rows) - 1))
        try:
            epi = self.valid_rows.iloc[probe_idx]["episode_index"]
            if np.isfinite(float(epi)):
                return int(float(epi))
        except Exception:
            return None
        return None

    def _maybe_handle_immediate_redo_request(self) -> bool:
        """Return True when an immediate redo was handled and this tick should skip normal replay."""
        if not self.recording_enabled:
            self._redo_prev_episode_event.clear()
            return False
        if not self._redo_prev_episode_event.is_set():
            return False

        episode_for_redo = self._resolve_current_episode_for_redo()
        self._redo_prev_episode_event.clear()
        if episode_for_redo is None:
            self.update_status("REDO", "left key pressed, but current episode is unknown")
            return False

        self._rewind_to_episode(episode_for_redo)
        self.current_episode_idx = episode_for_redo
        self._skip_reset_event.clear()
        self._is_reset_waiting = False
        return True

    def _build_episode_row_ranges(self) -> dict[int, tuple[int, int]]:
        ranges: dict[int, tuple[int, int]] = {}
        if "episode_index" not in self.valid_rows.columns:
            return ranges
        try:
            epi_series = pd.to_numeric(self.valid_rows["episode_index"], errors="coerce")
        except Exception:
            return ranges
        for i, epi in enumerate(epi_series.to_numpy()):
            if not np.isfinite(epi):
                continue
            e = int(epi)
            if e not in ranges:
                ranges[e] = (i, i + 1)
            else:
                s, _ = ranges[e]
                ranges[e] = (s, i + 1)
        return ranges

    def _ensure_tool_axis_columns(self):
        if {"tool_axis_x", "tool_axis_y", "tool_axis_z"}.issubset(self.df.columns):
            return
        missing = [c for c in ["tool_axis_x", "tool_axis_y", "tool_axis_z"] if c not in self.df.columns]
        raise ValueError(
            "CSV 缺少列: " + missing[0] +
            "。当前 replay 仅支持 tool_axis 版本 CSV。请先用最新 extract 流程重新导出 abs.csv。"
        )

    def _normalize_gripper_cmd_column_inplace(self):
        """Enforce unified CSV `gripper_cmd` semantic: [0, 1]."""
        if "gripper_cmd" not in self.df.columns:
            return
        series = pd.to_numeric(self.df["gripper_cmd"], errors="coerce")
        finite = series[np.isfinite(series.to_numpy(dtype=float))]
        if finite.empty:
            return

        gmin = float(finite.min())
        gmax = float(finite.max())
        if gmin < 0.0 or gmax > 1.0:
            raise ValueError(
                "CSV gripper_cmd must be normalized in [0,1]. "
                f"Found range [{gmin:.3f}, {gmax:.3f}]."
            )
        self.df["gripper_cmd"] = np.clip(series, 0.0, 1.0)
        print(f"[INFO] gripper_cmd validated in [0,1], range=[{gmin:.3f}, {gmax:.3f}]")

    def _build_bootstrap_transition(self, target_x, target_y, target_z, target_axis):
        if self.bootstrap_transition_steps <= 0:
            return
        self.bootstrap_queue = []
        current_transform = self.arm.fk(self.last_dof)
        start_x = float(current_transform[0, 3])
        start_y = float(current_transform[1, 3])
        start_z = float(current_transform[2, 3])
        start_axis = _normalize_axis(current_transform[:3, 2])
        for step_index in range(1, self.bootstrap_transition_steps + 1):
            alpha = float(step_index) / float(self.bootstrap_transition_steps)
            blended_x = (1.0 - alpha) * start_x + alpha * target_x
            blended_y = (1.0 - alpha) * start_y + alpha * target_y
            blended_z = (1.0 - alpha) * start_z + alpha * target_z
            blended_axis = _lerp_axis(start_axis, target_axis, alpha)
            self.bootstrap_queue.append((blended_x, blended_y, blended_z, blended_axis))

    def _limit_step(self, current_value, target_value, max_step):
        delta = target_value - current_value
        if delta > max_step:
            return current_value + max_step
        if delta < -max_step:
            return current_value - max_step
        return target_value

    def _apply_rate_limited_pose(self, target_x, target_y, target_z, target_axis, time_delta_s):
        position_step_limit = self.max_position_step_m
        axis_step_limit = self.max_axis_step_rad

        if time_delta_s > 0.0:
            scale = time_delta_s / 0.01
            position_step_limit *= scale
            axis_step_limit *= scale

        self.controller.x = self._limit_step(self.controller.x, target_x, position_step_limit)
        self.controller.y = self._limit_step(self.controller.y, target_y, position_step_limit)
        self.controller.z = self._limit_step(self.controller.z, target_z, position_step_limit)
        self.controller.tool_axis = _slerp_axis(self.controller.tool_axis, target_axis, axis_step_limit)

    def _controller_at_target(self) -> bool:
        pos = np.array([self.controller.x, self.controller.y, self.controller.z], dtype=float)
        target = np.array([self.controller.target_x, self.controller.target_y, self.controller.target_z], dtype=float)
        pos_err = float(np.linalg.norm(pos - target))
        axis_err = float(
            np.arccos(
                np.clip(
                    np.dot(
                        _normalize_axis(self.controller.tool_axis),
                        _normalize_axis(self.controller.target_tool_axis),
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        return pos_err < 0.003 and axis_err < math.radians(2.0)

    def _start_resync_to_target(self, target, reason_text: str) -> None:
        current_idx, x_value, y_value, z_value, ax, ay, az, gripper_cmd, episode_idx = target
        self.controller.apply_pose(x_value, y_value, z_value, ax, ay, az, gripper_cmd)
        self.resync_target = target
        self.update_status("RESYNC", f"row {current_idx}: {reason_text}, rate-limit to target", current_idx)
        print(
            f"[resync {current_idx:04d}] {reason_text}; "
            f"rate-limit toward xyz=({x_value:.3f},{y_value:.3f},{z_value:.3f}) "
            f"axis=({ax:.3f},{ay:.3f},{az:.3f})"
        )

    def update_status(self, state_text, last_msg="", last_csv_idx=None):
        with self.status_lock:
            self.current_status["row_idx"] = self.row_idx
            self.current_status["x"] = self.controller.x
            self.current_status["y"] = self.controller.y
            self.current_status["z"] = self.controller.z
            self.current_status["axis"] = self.controller.tool_axis.tolist()
            self.current_status["state"] = state_text
            self.current_status["mode"] = "AUTO" if self.auto_mode else "STEP"
            self.current_status["last_msg"] = last_msg
            if last_csv_idx is not None:
                self.current_status["last_csv_idx"] = last_csv_idx

    # ------------------------------------------------------------------
    # Target tuple format: (idx, x, y, z, ax, ay, az, gripper_cmd, episode_idx)
    # ------------------------------------------------------------------

    def _read_target_from_row(self, row, row_idx):
        """Read a single target from a CSV row."""
        x_value = float(row["mid_base_x"]) + self.target_offset_x
        y_value = float(row["mid_base_y"]) + self.target_offset_y
        z_value = float(row["mid_base_z"]) + self.target_offset_z
        ax = float(row["tool_axis_x"])
        ay = float(row["tool_axis_y"])
        az = float(row["tool_axis_z"])
        current_idx = int(row["abs_index"]) if "abs_index" in row.index else row_idx

        gripper_cmd = None
        if "gripper_cmd" in row.index:
            try:
                candidate = float(row["gripper_cmd"])
                if np.isfinite(candidate):
                    gripper_cmd = candidate
            except Exception:
                pass

        episode_idx = None
        if "episode_index" in row.index:
            try:
                epi = float(row["episode_index"])
                if np.isfinite(epi):
                    episode_idx = int(epi)
            except Exception:
                episode_idx = None

        return (current_idx, x_value, y_value, z_value, ax, ay, az, gripper_cmd, episode_idx)

    def _interpolate_targets(self, prev_target, curr_target, n_steps):
        """Generate interpolated targets between prev and curr using vector LERP."""
        _, px, py, pz, pax, pay, paz, pg, pep = prev_target
        ci, cx, cy, cz, cax, cay, caz, cg, cep = curr_target
        prev_axis = _normalize_axis(np.array([pax, pay, paz]))
        curr_axis = _normalize_axis(np.array([cax, cay, caz]))
        results = deque()
        for step in range(1, n_steps + 1):
            alpha = float(step) / float(n_steps)
            bx = (1.0 - alpha) * px + alpha * cx
            by = (1.0 - alpha) * py + alpha * cy
            bz = (1.0 - alpha) * pz + alpha * cz
            ba = _lerp_axis(prev_axis, curr_axis, alpha)
            if pg is None or cg is None:
                bg = cg if cg is not None else pg
            else:
                bg = (1.0 - alpha) * pg + alpha * cg
            synthetic_idx = prev_target[0] + step
            results.append((synthetic_idx, bx, by, bz, float(ba[0]), float(ba[1]), float(ba[2]), bg, cep if cep is not None else pep))
        return results

    def get_next_valid_target(self):
        if self.interpolated_target_queue:
            return self.interpolated_target_queue.popleft()

        invalid_count = 0
        while self.row_idx < len(self.valid_rows):
            row = self.valid_rows.iloc[self.row_idx]
            self.row_idx += 1
            target = self._read_target_from_row(row, self.row_idx - 1)
            current_idx, x_value, y_value, z_value, ax, ay, az, gripper_cmd, episode_idx = target

            if y_value > (self.soft_y_limit + 1e-6):
                if self.enable_soft_y_filter:
                    y_value = self.soft_y_limit
                    target = (current_idx, x_value, y_value, z_value, ax, ay, az, gripper_cmd, episode_idx)
                    self.update_status("CLAMP", f"clamp row {current_idx}: y -> {self.soft_y_limit:.3f}", current_idx)
                else:
                    self.update_status("SKIP", f"skip row {current_idx}: y={y_value:.3f} > {self.soft_y_limit:.3f}", current_idx)
                    invalid_count += 1
                    continue

            if self.previous_valid_target is not None and invalid_count > 0:
                interp = self._interpolate_targets(self.previous_valid_target, target, invalid_count + 1)
                self.interpolated_target_queue = interp
                self.update_status("INTERP", f"filled {invalid_count} invalid frames with interpolation", current_idx)
                print(f"[interp] filled {invalid_count} invalid frames before row {current_idx:04d}")
                return self.interpolated_target_queue.popleft()

            return target

        return None

    def _is_reachable_target(self, x_value, y_value, z_value, ax, ay, az):
        """Check if a target is IK-reachable without excessive joint jump."""
        x_value, y_value, z_value = self.controller.clamp_pose(x_value, y_value, z_value)
        p_target = np.array([x_value, y_value, z_value])
        axis_target = _normalize_axis(np.array([ax, ay, az]))

        dof, info = self.arm.ik(p_target, axis_target, current_arm_motor_q=self.last_dof)
        success = bool(info.get("success", False)) if isinstance(info, dict) else bool(info)
        if not success:
            return False, "ik_failed"

        candidate_dof = np.asarray(dof, dtype=float).reshape(-1)
        reference_dof = np.asarray(self.last_dof, dtype=float).reshape(-1)
        compare_count = min(5, candidate_dof.shape[0], reference_dof.shape[0])
        if compare_count <= 0:
            return False, "invalid_dof"
        jump_max = float(np.max(np.abs(candidate_dof[:compare_count] - reference_dof[:compare_count])))
        if jump_max > self.max_candidate_joint_jump_rad:
            return False, f"joint_jump>{self.max_candidate_joint_jump_rad:.2f} ({jump_max:.3f})"
        return True, "ok"

    def _rewind_to_episode(self, episode_idx: int):
        if episode_idx not in self.episode_row_ranges:
            print(f"[redo] episode {episode_idx} range not found; cannot rewind")
            return
        start_row, _end_row = self.episode_row_ranges[episode_idx]
        self.row_idx = int(start_row)
        self.interpolated_target_queue.clear()
        self.previous_valid_target = None
        self.resync_target = None
        self.bootstrap_queue = []
        self.has_applied_first_target = False
        self.last_emitted_abs_index = -1

        before = len(self.sent_actions_records)
        self.sent_actions_records = [
            r for r in self.sent_actions_records
            if int(r.get("episode_index", -1)) != int(episode_idx)
        ]
        removed = before - len(self.sent_actions_records)
        self.sent_action_seq = len(self.sent_actions_records)

        if self.video_recorder is not None:
            self.video_recorder.drop_episode_records(episode_idx)

        msg = f"redo episode {episode_idx}: rewind to row {start_row}, drop {removed} action rows"
        self.update_status("REDO", msg)
        print(f"[redo] {msg}")

    def _maybe_wait_for_episode_reset(self, next_episode_idx: Optional[int]) -> bool:
        if next_episode_idx is None:
            return True
        if self.current_episode_idx is None:
            self.current_episode_idx = next_episode_idx
            return True
        if next_episode_idx == self.current_episode_idx:
            return True

        prev = self.current_episode_idx
        self.current_episode_idx = next_episode_idx
        if self.episode_reset_time_s <= 0.0:
            return True

        wait_s = float(self.episode_reset_time_s)
        msg = f"episode {prev} -> {next_episode_idx}, wait {wait_s:.1f}s for reset"
        self.update_status("RESET_WAIT", msg)
        print(f"[reset] {msg}")
        self._is_reset_waiting = True
        self._skip_reset_event.clear()
        self._redo_prev_episode_event.clear()
        end_time = time.time() + wait_s
        while self.is_running and time.time() < end_time:
            if self._redo_prev_episode_event.is_set():
                self._rewind_to_episode(prev)
                self.current_episode_idx = prev
                self._redo_prev_episode_event.clear()
                self._skip_reset_event.clear()
                self._is_reset_waiting = False
                return False
            if self._skip_reset_event.is_set():
                print("[reset] skip requested by RIGHT key, continue to next episode now")
                break
            time.sleep(0.05)
        self._redo_prev_episode_event.clear()
        self._skip_reset_event.clear()
        self._is_reset_waiting = False
        return True

    def update_target_from_csv(self):
        if self._maybe_handle_immediate_redo_request():
            return

        if self.resync_target is not None:
            current_idx = int(self.resync_target[0])
            if self._controller_at_target():
                self.previous_valid_target = self.resync_target
                self.resync_target = None
                self.has_applied_first_target = True
                self.update_status("RUNNING", f"resync reached row {current_idx}", current_idx)
                print(f"[resync {current_idx:04d}] reached target, resume CSV replay")
            else:
                self.update_status("RESYNC", f"moving toward row {current_idx}", current_idx)
                return

        if self.row_idx >= len(self.valid_rows) and not self.interpolated_target_queue:
            self.update_status("FINISHED", "all rows finished")
            self.is_running = False
            return

        if not self.auto_mode:
            if not self.step_once:
                self.update_status("PAUSED", "waiting for next step")
                return
        else:
            now = time.time()
            if self.is_paused:
                self.update_status("PAUSED", "auto paused")
                return
            if now - self.last_frame_time < 1.0 / self.fps:
                return
            self.last_frame_time = now

        target = self.get_next_valid_target()
        self.step_once = False

        skipped_unreachable = 0
        first_skipped_target = None
        while target is not None:
            current_idx, x_value, y_value, z_value, ax, ay, az, gripper_cmd, episode_idx = target
            reachable, reason_text = self._is_reachable_target(x_value, y_value, z_value, ax, ay, az)
            if reachable:
                break
            if self.auto_resync_on_joint_jump and reason_text.startswith("joint_jump>"):
                if not self._maybe_wait_for_episode_reset(episode_idx):
                    return
                self.current_episode_idx = episode_idx if episode_idx is not None else self.current_episode_idx
                self._start_resync_to_target(target, reason_text)
                return
            if first_skipped_target is None:
                first_skipped_target = target
            skipped_unreachable += 1
            self.update_status("SKIP", f"skip row {current_idx}: {reason_text}", current_idx)
            print(
                f"[skip {current_idx:04d}] {reason_text} | "
                f"xyz=({x_value:.3f},{y_value:.3f},{z_value:.3f}) "
                f"axis=({ax:.3f},{ay:.3f},{az:.3f})"
            )
            target = self.get_next_valid_target()

        if target is None:
            self.update_status("FINISHED", "no more valid rows")
            self.is_running = False
            return

        current_idx, x_value, y_value, z_value, ax, ay, az, gripper_cmd, episode_idx = target

        if skipped_unreachable > 0 and self.previous_valid_target is not None and first_skipped_target is not None:
            repaired = self._interpolate_targets(self.previous_valid_target, target, skipped_unreachable + 1)
            self.interpolated_target_queue.extendleft(reversed(repaired))
            target = self.interpolated_target_queue.popleft()
            current_idx, x_value, y_value, z_value, ax, ay, az, gripper_cmd, episode_idx = target
            self.update_status("INTERP", f"filled {skipped_unreachable} unreachable frames", current_idx)
            print(f"[interp] filled {skipped_unreachable} unreachable frames")

        if not self._maybe_wait_for_episode_reset(episode_idx):
            return
        self.controller.apply_pose(x_value, y_value, z_value, ax, ay, az, gripper_cmd)
        self.previous_valid_target = target

        if not self.has_applied_first_target:
            self._build_bootstrap_transition(
                self.controller.target_x, self.controller.target_y, self.controller.target_z,
                self.controller.target_tool_axis,
            )
            self.has_applied_first_target = True

        axis_str = f"({ax:.3f},{ay:.3f},{az:.3f})"
        msg = f"row {current_idx}: x={x_value:.3f}, y={y_value:.3f}, z={z_value:.3f}, axis={axis_str}"
        self.update_status("RUNNING", msg, current_idx)
        print(
            f"[row {current_idx:04d}] "
            f"target=({x_value:.3f},{y_value:.3f},{z_value:.3f}), "
            f"axis={axis_str}, gripper={self.controller.gripper_pos:.2f}"
        )

    def robot_control_loop(self):
        profile_last_print_time = time.perf_counter()
        profile_count = 0
        profile_loop_sum = 0.0
        profile_ik_sum = 0.0
        profile_send_sum = 0.0
        profile_video_sum = 0.0
        profile_loop_max = 0.0
        profile_ik_max = 0.0
        profile_send_max = 0.0
        profile_video_max = 0.0

        while self.is_running:
            loop_start_time = time.perf_counter()
            if self._maybe_handle_immediate_redo_request():
                time.sleep(0.01)
                continue
            self.update_target_from_csv()

            abs_index_now = int(self.current_status.get("last_csv_idx", -1))
            motion_pending = bool(self.bootstrap_queue) or not self._controller_at_target()
            if abs_index_now == self.last_emitted_abs_index and not motion_pending:
                time.sleep(0.005)
                continue

            send_period_s = 1.0 / max(float(self.fps or 30.0), 1.0)
            now_wall = time.time()
            if now_wall - self.last_send_time < send_period_s:
                time.sleep(0.005)
                continue

            control_delta_s = now_wall - self.last_control_time if self.last_control_time else send_period_s
            self.last_control_time = now_wall
            self.last_send_time = now_wall

            if self.bootstrap_queue:
                next_pose = self.bootstrap_queue.pop(0)
                self.controller.x = next_pose[0]
                self.controller.y = next_pose[1]
                self.controller.z = next_pose[2]
                self.controller.tool_axis = next_pose[3].copy()
            else:
                self._apply_rate_limited_pose(
                    self.controller.target_x,
                    self.controller.target_y,
                    self.controller.target_z,
                    self.controller.target_tool_axis,
                    control_delta_s,
                )

            # 5DOF IK: position + tool axis direction
            mujoco.mj_forward(self.model, self.data)
            p_target = np.array([self.controller.x, self.controller.y, self.controller.z])
            axis_target = self.controller.tool_axis

            ik_start_time = time.perf_counter()
            dof, info = self.arm.ik(p_target, axis_target, current_arm_motor_q=self.last_dof)
            ik_dt = time.perf_counter() - ik_start_time
            success = info.get("success", False) if isinstance(info, dict) else bool(info)

            if dof is None or not success:
                self.update_status("IK_FAIL", "ik returned invalid result")
                time.sleep(0.01)
                continue

            self.last_dof = dof

            if len(dof) >= 6:
                dof[5] = self.controller.gripper_pos

            self.data.qpos[:6] = dof[:6]
            mujoco.mj_step(self.model, self.data)

            sim_deg = [math.degrees(q) for q in dof[:5]]
            real_pan = sim_deg[0] * -1.0 + 3
            real_lift = sim_deg[1] * 1.0 + 90
            real_elbow = sim_deg[2] * 1.0 - 90
            real_wrist = sim_deg[3] * 1.0 + 0
            real_roll = sim_deg[4] * 1.0 + 90

            action_dict = {
                "shoulder_pan.pos": real_pan,
                "shoulder_lift.pos": real_lift,
                "elbow_flex.pos": real_elbow,
                "wrist_flex.pos": real_wrist,
                "wrist_roll.pos": real_roll,
                "gripper.pos": np.clip(self.controller.gripper_pos * 100.0, 0.0, 100.0),
            }

            if self.real_robot is not None:
                send_start_time = time.perf_counter()
                self.real_robot.send_action(action_dict)
                send_dt = time.perf_counter() - send_start_time
            else:
                send_dt = 0.0
            video_frame_index = -1
            video_timestamp_s = float("nan")
            episode_idx_now = int(self.current_episode_idx) if self.current_episode_idx is not None else -1
            if self.video_recorder is not None:
                video_start_time = time.perf_counter()
                video_frame_index, video_timestamp_s = self.video_recorder.capture_one(
                    action_seq=self.sent_action_seq,
                    csv_row_index=int(self.current_status.get("row_idx", -1)),
                    abs_index=abs_index_now,
                    episode_index=episode_idx_now,
                )
                video_dt = time.perf_counter() - video_start_time
            else:
                video_dt = 0.0
            self.sent_actions_records.append(
                {
                    "action_seq": int(self.sent_action_seq),
                    "episode_index": int(episode_idx_now),
                    "frame_index": int(self.current_status.get("row_idx", -1)),
                    "csv_row_index": abs_index_now,
                    "abs_index": abs_index_now,
                    "shoulder_pan.pos": float(action_dict["shoulder_pan.pos"]),
                    "shoulder_lift.pos": float(action_dict["shoulder_lift.pos"]),
                    "elbow_flex.pos": float(action_dict["elbow_flex.pos"]),
                    "wrist_flex.pos": float(action_dict["wrist_flex.pos"]),
                    "wrist_roll.pos": float(action_dict["wrist_roll.pos"]),
                    "gripper.pos": float(action_dict["gripper.pos"]),
                    "video_frame_index": int(video_frame_index),
                    "video_timestamp_s": float(video_timestamp_s) if np.isfinite(video_timestamp_s) else np.nan,
                    "recorded_at_unix_s": float(time.time()),
                }
            )
            self.last_emitted_abs_index = abs_index_now
            self.sent_action_seq += 1

            if self.profile_control:
                loop_dt = time.perf_counter() - loop_start_time
                profile_count += 1
                profile_loop_sum += loop_dt
                profile_ik_sum += ik_dt
                profile_send_sum += send_dt
                profile_video_sum += video_dt
                profile_loop_max = max(profile_loop_max, loop_dt)
                profile_ik_max = max(profile_ik_max, ik_dt)
                profile_send_max = max(profile_send_max, send_dt)
                profile_video_max = max(profile_video_max, video_dt)
                now_perf = time.perf_counter()
                if now_perf - profile_last_print_time >= 1.0:
                    print(
                        "[PROFILE] "
                        f"loop avg/max={profile_loop_sum/profile_count*1000:.1f}/{profile_loop_max*1000:.1f}ms | "
                        f"ik avg/max={profile_ik_sum/profile_count*1000:.1f}/{profile_ik_max*1000:.1f}ms | "
                        f"send avg/max={profile_send_sum/profile_count*1000:.1f}/{profile_send_max*1000:.1f}ms | "
                        f"video avg/max={profile_video_sum/profile_count*1000:.1f}/{profile_video_max*1000:.1f}ms",
                        flush=True,
                    )
                    profile_last_print_time = now_perf
                    profile_count = 0
                    profile_loop_sum = 0.0
                    profile_ik_sum = 0.0
                    profile_send_sum = 0.0
                    profile_video_sum = 0.0
                    profile_loop_max = 0.0
                    profile_ik_max = 0.0
                    profile_send_max = 0.0
                    profile_video_max = 0.0

            time.sleep(0.001)

    def dump_sent_actions_if_needed(self):
        if self.export_action_csv is None:
            return
        self.export_action_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(self.sent_actions_records)
        df.to_csv(self.export_action_csv, index=False)
        print(f"[INFO] exported sent action csv: {self.export_action_csv} (rows={len(df)})")

    def draw_status_panel(self):
        canvas = np.zeros((300, 820, 3), dtype=np.uint8)
        with self.status_lock:
            status = self.current_status.copy()

        color = (0, 255, 0)
        if status["state"] == "PAUSED":
            color = (0, 255, 255)
        elif status["state"] == "FINISHED":
            color = (255, 255, 0)
        elif status["state"] in ["IK_FAIL", "SKIP"]:
            color = (0, 0, 255)

        def put(text, y, c=(255, 255, 255), scale=0.7):
            cv2.putText(canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv2.LINE_AA)

        put("SO100 CSV Replay", 35, (255, 255, 255), 0.9)
        put(f"state: {status['state']}", 75, color, 0.75)
        put(f"mode: {status['mode']}", 110, (255, 255, 255), 0.75)
        put(f"row pointer: {status['row_idx']} / {status['total_rows']}", 145, (255, 255, 255), 0.75)
        put(f"last csv idx: {status['last_csv_idx']}", 180, (255, 255, 255), 0.75)
        put(f"target xyz: {status['x']:.3f}, {status['y']:.3f}, {status['z']:.3f}", 215, (0, 255, 0), 0.75)
        ax = status.get("axis", [0, 0, -1])
        put(f"tool axis: {ax[0]:.3f}, {ax[1]:.3f}, {ax[2]:.3f}", 250, (0, 220, 255), 0.75)
        put(f"msg: {status['last_msg'][:95]}", 285, (200, 200, 200), 0.58)

        return canvas

    def run_status_loop(self):
        cv2.namedWindow("SO100 Replay Status", cv2.WINDOW_NORMAL)

        try:
            while self.is_running:
                panel = self.draw_status_panel()
                cv2.imshow("SO100 Replay Status", panel)
                key_raw = cv2.waitKeyEx(30)
                key = key_raw & 0xFF

                if key == ord(" "):
                    if not self.auto_mode:
                        self.step_once = True
                        self.update_status("STEP", "execute one frame")
                    else:
                        self.is_paused = not self.is_paused
                        self.update_status("PAUSED" if self.is_paused else "RUNNING", "toggle auto pause")
                elif key == ord("a"):
                    self.auto_mode = not self.auto_mode
                    if self.auto_mode:
                        self.is_paused = False
                        self.update_status("RUNNING", "switch to AUTO")
                    else:
                        self.is_paused = True
                        self.update_status("PAUSED", "switch to STEP")
                elif key == ord("p"):
                    if self.auto_mode:
                        self.is_paused = not self.is_paused
                        self.update_status("PAUSED" if self.is_paused else "RUNNING", "toggle auto pause")
                elif key == ord("q") or key == 27:
                    self.is_running = False
                    break
                # Left arrow: immediate redo of current episode (recording enabled only).
                # Common key codes: Linux/X11 65361, OpenCV waitKeyEx 2424832, some backends 81.
                elif key_raw in (65361, 2424832) or key == 81:
                    if self.recording_enabled:
                        self._redo_prev_episode_event.set()
                        self.update_status("REDO", "redo current episode by LEFT key")
                # Right arrow: when waiting reset, skip remaining wait and continue immediately.
                # Common key codes: Linux/X11 65363, OpenCV waitKeyEx 2555904, some backends 83.
                elif key_raw in (65363, 2555904) or key == 83:
                    if self._is_reset_waiting:
                        self._skip_reset_event.set()
                        self.update_status("RESET_WAIT", "skip reset wait by RIGHT key")

            final_panel = self.draw_status_panel()
            cv2.imshow("SO100 Replay Status", final_panel)
            cv2.waitKey(300)
        except KeyboardInterrupt:
            self.is_running = False
        finally:
            cv2.destroyAllWindows()


def _parse_replay_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--target-offset-x", type=float, default=REPLAY_TARGET_OFFSET_DEFAULT_X)
    parser.add_argument("--target-offset-y", type=float, default=REPLAY_TARGET_OFFSET_DEFAULT_Y)
    parser.add_argument("--target-offset-z", type=float, default=REPLAY_TARGET_OFFSET_DEFAULT_Z)
    parser.add_argument("--export-action-csv", default=None, help="Optional path to export actual sent actions.")
    parser.add_argument("--record-video-path", default=None, help="Optional MP4 path to record replay camera video.")
    parser.add_argument("--record-rs-serial", default=None, help="Optional RealSense serial for replay recording.")
    parser.add_argument("--record-video-width", type=int, default=640, help="Replay recording width.")
    parser.add_argument("--record-video-height", type=int, default=480, help="Replay recording height.")
    parser.add_argument("--record-video-fps", type=float, default=30.0, help="Replay recording FPS.")
    parser.add_argument("--episode-reset-time-s", type=float, default=0.0, help="Pause duration in seconds between episode_index boundaries.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing sent_action/map exports.")
    parser.add_argument("--profile-control", action="store_true", help="Print replay loop timing diagnostics once per second.")
    parser.add_argument(
        "--no-auto-resync-on-jump",
        action="store_true",
        help="Keep old behavior: skip targets whose IK candidate has a large joint jump.",
    )
    return parser.parse_known_args()


def _normalize_replay_video_output_path(record_video_path: Optional[str], csv_path: str) -> Optional[str]:
    """
    Auto organize replay video outputs under:
      .../replay_videos/<run_name>/<original_file_name>.mp4
    """
    if not record_video_path:
        return record_video_path
    p = Path(record_video_path).expanduser()
    parent = p.parent
    if parent.name != "replay_videos":
        return str(p)

    stem = p.stem
    run_name = stem[:-6] if stem.endswith("_robot") else stem
    if not run_name:
        run_name = Path(csv_path).stem.replace("_d435_chunk-000_file-000_abs", "")
        if not run_name:
            run_name = "replay_run"

    out = parent / run_name / p.name
    return str(out)


def main():
    arguments, _unknown = _parse_replay_args()
    print(
        "[INFO] replay threading: "
        f"OMP={os.environ.get('OMP_NUM_THREADS')} "
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS')}"
    )

    csv_path = arguments.csv_path
    port = arguments.port
    fps = arguments.fps
    target_offset_x = float(arguments.target_offset_x)
    target_offset_y = float(arguments.target_offset_y)
    target_offset_z = float(arguments.target_offset_z)
    export_action_csv = arguments.export_action_csv
    record_video_path = arguments.record_video_path
    record_rs_serial = arguments.record_rs_serial
    record_video_width = int(arguments.record_video_width)
    record_video_height = int(arguments.record_video_height)
    record_video_fps = float(arguments.record_video_fps)
    episode_reset_time_s = max(0.0, float(arguments.episode_reset_time_s))
    resume = bool(arguments.resume)
    profile_control = bool(arguments.profile_control)
    auto_resync_on_joint_jump = not bool(arguments.no_auto_resync_on_jump)

    if not csv_path:
        csv_path = input("请输入轨迹 CSV 路径: ").strip()
    record_video_path = _normalize_replay_video_output_path(record_video_path, csv_path)
    if record_video_path:
        print(f"[INFO] normalized replay video output: {record_video_path}")
    has_port = port is not None and str(port).strip() != ""
    export_only = bool(export_action_csv) and (not has_port)
    if not export_only and not port:
        port = input("请输入 USB 端口 (回车默认 /dev/ttyACM0): ").strip() or "/dev/ttyACM0"
    if fps is None:
        fps_str = input("请输入自动回放 FPS (回车默认 5): ").strip()
        fps = float(fps_str) if fps_str else 5.0

    real_robot = None
    if not export_only:
        robot_config = SO100FollowerConfig(port=port, id="single_arm", use_degrees=True)
        real_robot = SO100Follower(robot_config)
        real_robot.connect()

    app = TrajectoryReplayApp(
        csv_path=csv_path,
        real_robot=real_robot,
        fps=fps,
        target_offset_x=target_offset_x,
        target_offset_y=target_offset_y,
        target_offset_z=target_offset_z,
        export_action_csv=export_action_csv,
        record_video_path=record_video_path,
        record_rs_serial=record_rs_serial,
        record_video_width=record_video_width,
        record_video_height=record_video_height,
        record_video_fps=record_video_fps,
        episode_reset_time_s=episode_reset_time_s,
        resume=resume,
        profile_control=profile_control,
        auto_resync_on_joint_jump=auto_resync_on_joint_jump,
    )
    try:
        if export_only:
            app.auto_mode = True
            app.is_paused = False
            print("[INFO] export-only mode: robot disconnected, running headless replay for action export...")
            while app.is_running:
                time.sleep(0.02)
        else:
            app.run_status_loop()
    finally:
        app.is_running = False
        app.robot_thread.join()
        app.dump_sent_actions_if_needed()
        if app.video_recorder is not None:
            app.video_recorder.stop()
        if real_robot is not None:
            real_robot.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
