#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_alpha_series(length: int, alpha_max: float, profile: str) -> np.ndarray:
    if length <= 1:
        return np.asarray([alpha_max], dtype=float)
    linear = np.linspace(0.0, 1.0, num=length, dtype=float)
    if profile == "cosine":
        ramp = 0.5 - 0.5 * np.cos(np.pi * linear)
    else:
        ramp = linear
    return alpha_max * ramp


def blend_series(
    original_series: np.ndarray,
    anchor_value: float,
    alpha_series: np.ndarray,
) -> np.ndarray:
    return (1.0 - alpha_series) * original_series + alpha_series * anchor_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse GraspNet anchor into HaMeR trajectory window.")
    parser.add_argument("--input-csv", required=True, help="smoothed trajectory csv")
    parser.add_argument("--grasp-prior-json", required=True, help="top_grasp json from grasp pipeline")
    parser.add_argument("--grasp-window-json", required=True, help="window json from window detector")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--alpha-max", type=float, default=0.45, help="max blending weight in window")
    parser.add_argument("--alpha-profile", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--min-score-to-fuse", type=float, default=0.1, help="if grasp score below this, skip fusion")
    parser.add_argument("--overwrite-orientation", action="store_true", help="if set, blend base roll/pitch/yaw to grasp rotation")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input_csv)
    required_xyz = ["mid_base_x", "mid_base_y", "mid_base_z"]
    for column_name in required_xyz:
        if column_name not in dataframe.columns:
            raise ValueError(f"Missing required column: {column_name}")

    grasp_prior = load_json(args.grasp_prior_json)
    grasp_window = load_json(args.grasp_window_json)

    window = grasp_window["window"]
    start_index = int(window["start_index"])
    end_index = int(window["end_index"])
    if start_index < 0 or end_index >= len(dataframe) or end_index < start_index:
        raise ValueError("Invalid grasp window indices.")

    top_grasp = grasp_prior["top_grasp"]
    grasp_score = float(top_grasp["score"])
    anchor_xyz_base = np.asarray(top_grasp["translation_base_xyz"], dtype=float).reshape(3)
    if grasp_score < args.min_score_to_fuse:
        dataframe["grasp_fused"] = 0
        dataframe["grasp_alpha"] = 0.0
        output_path = Path(args.output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        print(f"[INFO] Grasp score {grasp_score:.4f} < min-score-to-fuse, skipped fusion.")
        print(f"[INFO] Output CSV: {output_path}")
        return

    fused_dataframe = dataframe.copy()
    fused_dataframe["grasp_fused"] = 0
    fused_dataframe["grasp_alpha"] = 0.0

    window_indices = np.arange(start_index, end_index + 1, dtype=int)
    alpha_series = build_alpha_series(
        length=len(window_indices),
        alpha_max=float(args.alpha_max),
        profile=args.alpha_profile,
    )

    for axis_offset, axis_name in enumerate(["mid_base_x", "mid_base_y", "mid_base_z"]):
        original_series = fused_dataframe.loc[window_indices, axis_name].to_numpy(dtype=float)
        fused_series = blend_series(
            original_series=original_series,
            anchor_value=float(anchor_xyz_base[axis_offset]),
            alpha_series=alpha_series,
        )
        fused_dataframe.loc[window_indices, axis_name] = fused_series

    if args.overwrite_orientation and {"base_roll", "base_pitch", "base_yaw"}.issubset(fused_dataframe.columns):
        rotation_matrix = np.asarray(top_grasp["rotation_base_3x3"], dtype=float).reshape(3, 3)
        pitch = -np.arcsin(rotation_matrix[2, 0])
        if abs(rotation_matrix[2, 0]) < 0.999999:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
        for column_name, anchor_value in [
            ("base_roll", float(roll)),
            ("base_pitch", float(pitch)),
            ("base_yaw", float(yaw)),
        ]:
            original_series = fused_dataframe.loc[window_indices, column_name].to_numpy(dtype=float)
            fused_series = blend_series(
                original_series=original_series,
                anchor_value=anchor_value,
                alpha_series=alpha_series,
            )
            fused_dataframe.loc[window_indices, column_name] = fused_series

    fused_dataframe.loc[window_indices, "grasp_fused"] = 1
    fused_dataframe.loc[window_indices, "grasp_alpha"] = alpha_series

    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused_dataframe.to_csv(output_path, index=False)

    print(f"[INFO] Fused grasp anchor into window [{start_index}, {end_index}]")
    print(f"[INFO] Grasp score: {grasp_score:.4f}")
    print(f"[INFO] Output CSV: {output_path}")


if __name__ == "__main__":
    main()
