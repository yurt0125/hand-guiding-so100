"""Unified trajectory logger for observation/action/timestamp records."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common_control import ControlStepResult


def _target_to_vector(target) -> list[float]:
    return [
        float(target.x),
        float(target.y),
        float(target.z),
        float(target.tool_axis_x),
        float(target.tool_axis_y),
        float(target.tool_axis_z),
        float(target.gripper),
    ]


class TrajectoryLogger:
    """Write replay/control transitions as JSONL rows."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_index = 0
        self.previous_observation_vector: list[float] | None = None
        self.previous_timestamp_s: float | None = None

    def log_step(self, result: ControlStepResult) -> None:
        timestamp_s = float(result.raw_target.timestamp if result.raw_target.timestamp is not None else time.time())
        observation_vector = _target_to_vector(result.safe_target)

        action_rel = None
        delta_t_s = None
        if self.previous_observation_vector is not None:
            action_rel = (
                np.asarray(observation_vector, dtype=float)
                - np.asarray(self.previous_observation_vector, dtype=float)
            ).tolist()
            if self.previous_timestamp_s is not None:
                delta_t_s = float(timestamp_s - self.previous_timestamp_s)

        payload: dict[str, Any] = {
            "sample_index": int(self.sample_index),
            "timestamp_s": timestamp_s,
            "delta_t_s": delta_t_s,
            "source": str(result.raw_target.source),
            "observation": {
                "safe_target_vector": observation_vector,
                "raw_target_vector": _target_to_vector(result.raw_target),
                "joint_deg_cmd": {key: float(value) for key, value in result.joint_deg_cmd.items()},
            },
            "action": {
                "relative_safe_target_vector": action_rel,
            },
            "runtime": {
                "ik_ok": bool(result.ik_ok),
                "mapping_quality": {key: float(value) for key, value in result.mapping_quality.items()},
            },
        }

        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self.previous_observation_vector = observation_vector
        self.previous_timestamp_s = timestamp_s
        self.sample_index += 1
