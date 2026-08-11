#!/usr/bin/env python3
"""Replay trajectory CSV in MuJoCo using the same control pipeline as real replay."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import pandas as pd

from common.common_control import CommonRunner
from common.common_robot import ACTIVE_ARM_JOINT_NAMES, EndEffectorTarget, SO100Kinematics
from common.config import (
    CSV_REPLAY_WORKSPACE,
    DEFAULT_CONTROL_GAIN,
    DEFAULT_FILTER_BACKEND,
    REPLAY_Y_MAX_THRESHOLD,
)


def _normalize_axis(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return a / norm


@dataclass
class ReplayRow:
    x: float
    y: float
    z: float
    tool_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    source_index: int = 0


class TrajectoryTable:
    REQUIRED_COLUMNS = ["mid_base_x", "mid_base_y", "mid_base_z", "tool_axis_x", "tool_axis_y", "tool_axis_z"]

    def __init__(self, csv_path: Path, keep_hand_found_only: bool):
        dataframe = pd.read_csv(csv_path)
        for column_name in self.REQUIRED_COLUMNS:
            if column_name not in dataframe.columns:
                raise KeyError(f"Missing required column: {column_name}")
        if keep_hand_found_only and "hand_found" in dataframe.columns:
            dataframe = dataframe[dataframe["hand_found"] == 1].copy()
        dataframe = dataframe[dataframe["mid_base_y"] <= REPLAY_Y_MAX_THRESHOLD].copy()
        dataframe = dataframe.reset_index(drop=True)
        if dataframe.empty:
            raise ValueError("No valid rows after filtering.")
        self.dataframe = dataframe

    def iter_rows(self):
        for row_index, row in self.dataframe.iterrows():
            axis = _normalize_axis(np.array([
                float(row["tool_axis_x"]),
                float(row["tool_axis_y"]),
                float(row["tool_axis_z"]),
            ]))
            yield ReplayRow(
                x=float(row["mid_base_x"]),
                y=float(row["mid_base_y"]),
                z=float(row["mid_base_z"]),
                tool_axis=axis,
                source_index=int(row_index),
            )


class SimSO100Hardware:
    """Minimal simulation hardware bridge implementing methods used by CommonRunner."""

    def __init__(self, kinematics: SO100Kinematics):
        self.kinematics = kinematics
        self.model = kinematics.model
        self.data = kinematics.data
        self.joint_count = len(ACTIVE_ARM_JOINT_NAMES)

    def get_joint_math_deg(self):
        output = {}
        for joint_index, joint_name in enumerate(ACTIVE_ARM_JOINT_NAMES):
            output[joint_name] = float(np.rad2deg(self.data.qpos[joint_index]))
        output["gripper"] = float(self.data.qpos[self.joint_count]) if self.model.nq > self.joint_count else 0.0
        return output

    def get_qpos_rad(self, nq: int):
        return self.data.qpos[:nq].copy()

    def send_math_deg_action(self, joint_math_deg, gripper_cmd: float):
        for joint_index, joint_name in enumerate(ACTIVE_ARM_JOINT_NAMES):
            self.data.qpos[joint_index] = np.deg2rad(float(joint_math_deg[joint_name]))
        if self.model.nq > self.joint_count:
            self.data.qpos[self.joint_count] = float(gripper_cmd)
        mujoco.mj_forward(self.model, self.data)


class ReplayTargetSource:
    def __init__(self, init_target: EndEffectorTarget, rows: list[ReplayRow]):
        self.target = EndEffectorTarget(**init_target.__dict__)
        self.rows = rows
        self.row_index = 0

    def next_target(self) -> Optional[EndEffectorTarget]:
        if self.row_index >= len(self.rows):
            return None
        row = self.rows[self.row_index]
        self.row_index += 1
        self.target.x = row.x
        self.target.y = row.y
        self.target.z = row.z
        self.target.tool_axis_x = float(row.tool_axis[0])
        self.target.tool_axis_y = float(row.tool_axis[1])
        self.target.tool_axis_z = float(row.tool_axis[2])
        self.target.source = f"csv_row_{row.source_index}"
        self.target.timestamp = time.time()
        return EndEffectorTarget(**self.target.__dict__)


def clamp_target_to_workspace(target: EndEffectorTarget) -> EndEffectorTarget:
    target.x = float(np.clip(target.x, CSV_REPLAY_WORKSPACE["x_min"], CSV_REPLAY_WORKSPACE["x_max"]))
    target.y = float(np.clip(target.y, CSV_REPLAY_WORKSPACE["y_min"], CSV_REPLAY_WORKSPACE["y_max"]))
    target.z = float(np.clip(target.z, CSV_REPLAY_WORKSPACE["z_min"], CSV_REPLAY_WORKSPACE["z_max"]))
    return target


def initialize_from_first_target(
    kinematics: SO100Kinematics,
    rows: list[ReplayRow],
) -> None:
    if not rows:
        return
    first_row = rows[0]
    p_target = np.array([
        float(np.clip(first_row.x, CSV_REPLAY_WORKSPACE["x_min"], CSV_REPLAY_WORKSPACE["x_max"])),
        float(np.clip(first_row.y, CSV_REPLAY_WORKSPACE["y_min"], CSV_REPLAY_WORKSPACE["y_max"])),
        float(np.clip(first_row.z, CSV_REPLAY_WORKSPACE["z_min"], CSV_REPLAY_WORKSPACE["z_max"])),
    ])
    axis_target = first_row.tool_axis
    current_q = kinematics.data.qpos[: kinematics.arm.model.nq].copy()
    try:
        dof, success = kinematics.ik(p_target, axis_target, current_q)
        if success:
            nq = min(len(dof), kinematics.model.nq)
            kinematics.data.qpos[:nq] = dof[:nq]
            kinematics.last_dof = dof.copy()
            mujoco.mj_forward(kinematics.model, kinematics.data)
    except Exception:
        pass


def render_replay_video(
    output_video_path: Path,
    kinematics: SO100Kinematics,
    runner: CommonRunner,
    source: ReplayTargetSource,
    fps: float,
    width: int,
    height: int,
) -> int:
    renderer = mujoco.Renderer(kinematics.model, width=width, height=height)
    try:
        import imageio.v2 as imageio  # type: ignore
    except ImportError as exc:
        raise ImportError("imageio is required to export mp4. Please install imageio.") from exc
    writer = imageio.get_writer(str(output_video_path), fps=fps, codec="libx264")

    frame_count = 0
    try:
        while True:
            target = source.next_target()
            if target is None:
                break
            target = clamp_target_to_workspace(target)
            runner.step(target)
            mujoco.mj_forward(kinematics.model, kinematics.data)
            renderer.update_scene(kinematics.data)
            writer.append_data(renderer.render())
            frame_count += 1
    finally:
        writer.close()
        renderer.close()
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay CSV trajectory in MuJoCo with CommonRunner logic.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--keep-hand-found-only", action="store_true", default=True)
    parser.add_argument("--initialize-from-first-target", action="store_true", default=True)
    args = parser.parse_args()

    input_csv_path = Path(args.input_csv).expanduser().resolve()
    output_video_path = Path(args.output_video).expanduser().resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    table = TrajectoryTable(input_csv_path, keep_hand_found_only=bool(args.keep_hand_found_only))
    rows = list(table.iter_rows())
    if len(rows) == 0:
        raise ValueError("No replay rows.")

    kinematics = SO100Kinematics()
    if args.initialize_from_first_target:
        initialize_from_first_target(kinematics, rows)
    hardware = SimSO100Hardware(kinematics)
    runner = CommonRunner(
        kin=kinematics,
        hardware=hardware,
        workspace=CSV_REPLAY_WORKSPACE,
        kp=DEFAULT_CONTROL_GAIN,
        filter_backend=DEFAULT_FILTER_BACKEND,
    )
    init_target = runner.initialize_from_robot()
    source = ReplayTargetSource(init_target=init_target, rows=rows)

    replayed_frames = render_replay_video(
        output_video_path=output_video_path,
        kinematics=kinematics,
        runner=runner,
        source=source,
        fps=float(args.fps),
        width=int(args.width),
        height=int(args.height),
    )

    print(f"[INFO] Replayed rows: {replayed_frames}")
    print(f"[INFO] Output video: {output_video_path}")


if __name__ == "__main__":
    main()
