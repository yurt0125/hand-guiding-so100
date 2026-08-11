#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_pinch_distance(dataframe: pd.DataFrame) -> np.ndarray:
    kp02 = dataframe[["kp02_cam_x", "kp02_cam_y", "kp02_cam_z"]].to_numpy(dtype=float)
    kp05 = dataframe[["kp05_cam_x", "kp05_cam_y", "kp05_cam_z"]].to_numpy(dtype=float)
    return np.linalg.norm(kp02 - kp05, axis=1)


def compute_pinch_velocity(distance: np.ndarray, timestamp_s: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(distance, dtype=float)
    if len(distance) <= 1:
        return velocity
    delta_distance = np.diff(distance)
    delta_time = np.diff(timestamp_s)
    safe_delta_time = np.where(np.abs(delta_time) < 1e-6, 1e-6, delta_time)
    velocity[1:] = np.abs(delta_distance / safe_delta_time)
    velocity[0] = velocity[1]
    return velocity


def find_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if (not value) and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def dilate_segment(start: int, end: int, total_length: int, margin: int) -> tuple[int, int]:
    return max(0, start - margin), min(total_length - 1, end + margin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect grasp window from HaMeR trajectory CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-scored-csv", default=None)
    parser.add_argument("--distance-threshold-m", type=float, default=0.035)
    parser.add_argument("--velocity-threshold-mps", type=float, default=0.08)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=2.0, help="score >= threshold enters candidate mask")
    parser.add_argument("--min-segment-frames", type=int, default=5)
    parser.add_argument("--dilation-frames", type=int, default=4)
    parser.add_argument("--fallback-last-frames", type=int, default=24)
    parser.add_argument("--prefer-last-segment", action="store_true", default=True)
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input_csv)
    required_columns = [
        "hand_found",
        "kp02_cam_x",
        "kp02_cam_y",
        "kp02_cam_z",
        "kp05_cam_x",
        "kp05_cam_y",
        "kp05_cam_z",
        "kp02_conf",
        "kp05_conf",
    ]
    for column_name in required_columns:
        if column_name not in dataframe.columns:
            raise ValueError(f"Missing required column: {column_name}")

    if "timestamp_s" in dataframe.columns:
        timestamp_s = dataframe["timestamp_s"].to_numpy(dtype=float)
    elif "t" in dataframe.columns:
        timestamp_s = dataframe["t"].to_numpy(dtype=float)
    else:
        timestamp_s = np.arange(len(dataframe), dtype=float)

    hand_found = dataframe["hand_found"].to_numpy(dtype=int) == 1
    pinch_distance_m = compute_pinch_distance(dataframe)
    pinch_velocity_mps = compute_pinch_velocity(pinch_distance_m, timestamp_s)
    min_confidence = np.minimum(
        dataframe["kp02_conf"].to_numpy(dtype=float),
        dataframe["kp05_conf"].to_numpy(dtype=float),
    )

    distance_hit = pinch_distance_m < args.distance_threshold_m
    velocity_hit = pinch_velocity_mps < args.velocity_threshold_mps
    confidence_low = min_confidence < args.confidence_threshold

    grasp_score = (
        distance_hit.astype(float)
        + velocity_hit.astype(float)
        + confidence_low.astype(float)
    )
    candidate_mask = hand_found & (grasp_score >= args.score_threshold)
    segments = find_segments(candidate_mask)
    segments = [
        (start, end)
        for start, end in segments
        if (end - start + 1) >= args.min_segment_frames
    ]

    selected_reason = "fallback_last_frames"
    total_frames = len(dataframe)
    if segments:
        selected_reason = "score_segment"
        selected_segment = segments[-1] if args.prefer_last_segment else segments[0]
        start_index, end_index = dilate_segment(
            selected_segment[0],
            selected_segment[1],
            total_frames,
            args.dilation_frames,
        )
    else:
        end_index = total_frames - 1
        start_index = max(0, total_frames - int(args.fallback_last_frames))

    output_payload: dict[str, Any] = {
        "source_csv": str(Path(args.input_csv).expanduser().resolve()),
        "window": {
            "start_index": int(start_index),
            "end_index": int(end_index),
            "length": int(end_index - start_index + 1),
            "reason": selected_reason,
        },
        "thresholds": {
            "distance_threshold_m": float(args.distance_threshold_m),
            "velocity_threshold_mps": float(args.velocity_threshold_mps),
            "confidence_threshold": float(args.confidence_threshold),
            "score_threshold": float(args.score_threshold),
            "min_segment_frames": int(args.min_segment_frames),
            "dilation_frames": int(args.dilation_frames),
            "fallback_last_frames": int(args.fallback_last_frames),
        },
        "segment_count": int(len(segments)),
        "segments": [{"start_index": int(s), "end_index": int(e), "length": int(e - s + 1)} for s, e in segments],
    }

    output_json_path = Path(args.output_json).expanduser().resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    scored_dataframe = dataframe.copy()
    scored_dataframe["pinch_distance_m"] = pinch_distance_m
    scored_dataframe["pinch_velocity_mps"] = pinch_velocity_mps
    scored_dataframe["min_pinch_conf"] = min_confidence
    scored_dataframe["distance_hit"] = distance_hit.astype(int)
    scored_dataframe["velocity_hit"] = velocity_hit.astype(int)
    scored_dataframe["confidence_low"] = confidence_low.astype(int)
    scored_dataframe["grasp_score"] = grasp_score
    scored_dataframe["grasp_candidate"] = candidate_mask.astype(int)

    output_scored_csv = args.output_scored_csv
    if output_scored_csv is None:
        output_scored_csv = str(output_json_path.with_suffix(".scored.csv"))
    scored_csv_path = Path(output_scored_csv).expanduser().resolve()
    scored_csv_path.parent.mkdir(parents=True, exist_ok=True)
    scored_dataframe.to_csv(scored_csv_path, index=False)

    print(f"[INFO] Grasp window JSON: {output_json_path}")
    print(f"[INFO] Scored CSV: {scored_csv_path}")
    print(f"[INFO] Window: [{start_index}, {end_index}] ({selected_reason})")


if __name__ == "__main__":
    main()
