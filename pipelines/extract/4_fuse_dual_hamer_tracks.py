#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_sync_indices(sync_map_jsonl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d435_indices: list[int] = []
    top_indices: list[int] = []
    with sync_map_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line.strip())
            if not row.get("within_threshold", True):
                continue
            d435_indices.append(int(row["idx_d435"]))
            top_indices.append(int(row["idx_top"]))
    if len(d435_indices) == 0:
        raise ValueError(f"No valid sync pairs in {sync_map_jsonl_path}")
    return np.asarray(d435_indices, dtype=int), np.asarray(top_indices, dtype=int)


def pick_value_by_confidence(
    rear_row: pd.Series,
    top_row: pd.Series,
    rear_conf_key: str,
    top_conf_key: str,
    rear_value_key: str,
    top_value_key: str,
    confidence_threshold: float,
):
    rear_conf = float(rear_row.get(rear_conf_key, np.nan))
    top_conf = float(top_row.get(top_conf_key, np.nan))
    rear_value = rear_row.get(rear_value_key, np.nan)
    top_value = top_row.get(top_value_key, np.nan)

    rear_ok = np.isfinite(rear_conf) and rear_conf >= confidence_threshold and np.isfinite(rear_value)
    top_ok = np.isfinite(top_conf) and top_conf >= confidence_threshold and np.isfinite(top_value)

    if rear_ok and top_ok:
        rear_weight = max(rear_conf, 1e-6)
        top_weight = max(top_conf, 1e-6)
        return float((rear_weight * float(rear_value) + top_weight * float(top_value)) / (rear_weight + top_weight)), "blend"
    if rear_ok:
        return float(rear_value), "rear"
    if top_ok:
        return float(top_value), "top"
    return np.nan, "none"


def main():
    parser = argparse.ArgumentParser(description="Fuse rear/top HaMeR tracks by sync map and confidence gating.")
    parser.add_argument("--rear-csv", required=True, help="Rear camera trajectory CSV (D435 side).")
    parser.add_argument("--top-csv", required=True, help="Top camera trajectory CSV.")
    parser.add_argument("--sync-map-jsonl", required=True, help="Sync map built from dual camera timestamps.")
    parser.add_argument("--output-csv", required=True, help="Output fused trajectory CSV.")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Keypoint confidence threshold.")
    args = parser.parse_args()

    rear_df = pd.read_csv(args.rear_csv)
    top_df = pd.read_csv(args.top_csv)
    d435_indices, top_indices = load_sync_indices(Path(args.sync_map_jsonl).expanduser().resolve())

    fused_rows: list[dict] = []
    used_rear = 0
    used_top = 0
    used_blend = 0

    for d435_idx, top_idx in zip(d435_indices, top_indices):
        if d435_idx < 0 or d435_idx >= len(rear_df) or top_idx < 0 or top_idx >= len(top_df):
            continue
        rear_row = rear_df.iloc[d435_idx]
        top_row = top_df.iloc[top_idx]
        output_row = dict(rear_row)
        output_row["top_source_row_index"] = int(top_idx)

        for keypoint_name in ["kp02", "kp05"]:
            for axis_name in ["u", "v", "cam_x", "cam_y", "cam_z"]:
                rear_key = f"{keypoint_name}_{axis_name}"
                top_key = f"{keypoint_name}_{axis_name}"
                conf_key = f"{keypoint_name}_conf"
                fused_value, source = pick_value_by_confidence(
                    rear_row=rear_row,
                    top_row=top_row,
                    rear_conf_key=conf_key,
                    top_conf_key=conf_key,
                    rear_value_key=rear_key,
                    top_value_key=top_key,
                    confidence_threshold=float(args.confidence_threshold),
                )
                output_row[f"{keypoint_name}_{axis_name}"] = fused_value
                output_row[f"{keypoint_name}_{axis_name}_source"] = source
                if source == "rear":
                    used_rear += 1
                elif source == "top":
                    used_top += 1
                elif source == "blend":
                    used_blend += 1

        if np.isfinite(output_row["kp02_cam_x"]) and np.isfinite(output_row["kp05_cam_x"]):
            midpoint_cam = 0.5 * np.asarray(
                [
                    [output_row["kp02_cam_x"], output_row["kp02_cam_y"], output_row["kp02_cam_z"]],
                    [output_row["kp05_cam_x"], output_row["kp05_cam_y"], output_row["kp05_cam_z"]],
                ],
                dtype=float,
            ).sum(axis=0)
            output_row["mid_cam_x"] = float(midpoint_cam[0])
            output_row["mid_cam_y"] = float(midpoint_cam[1])
            output_row["mid_cam_z"] = float(midpoint_cam[2])

        fused_rows.append(output_row)

    if len(fused_rows) == 0:
        raise ValueError("No fused rows produced. Check sync map and input CSV lengths.")

    output_df = pd.DataFrame(fused_rows)
    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"[INFO] Rear rows: {len(rear_df)}")
    print(f"[INFO] Top rows: {len(top_df)}")
    print(f"[INFO] Fused rows: {len(output_df)}")
    print(f"[INFO] Source counts - rear: {used_rear}, top: {used_top}, blend: {used_blend}")
    print(f"[INFO] Wrote: {output_path}")


if __name__ == "__main__":
    main()

