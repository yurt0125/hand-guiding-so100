#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyncPair:
    index_d435: int
    index_top: int
    timestamp_d435: float
    timestamp_top: float
    delta_top_to_d435_s: float
    within_threshold: bool


def load_npy_timestamps(path: Path) -> np.ndarray:
    values = np.load(str(path))
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Empty timestamps: {path}")
    return values


def load_jsonl_timestamps(path: Path, key: str = "timestamp_s") -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if key in payload:
                values.append(float(payload[key]))
    if len(values) == 0:
        raise ValueError(f"No '{key}' in jsonl: {path}")
    return np.asarray(values, dtype=float)


def load_top_timestamps(top_timestamps_path: Path | None, top_frame_meta_path: Path | None) -> np.ndarray:
    if top_timestamps_path is not None:
        return load_npy_timestamps(top_timestamps_path)
    if top_frame_meta_path is not None:
        return load_jsonl_timestamps(top_frame_meta_path, key="timestamp_s")
    raise ValueError("Provide either --top-timestamps or --top-frame-meta")


def find_nearest_index(sorted_timestamps: np.ndarray, target_timestamp: float) -> int:
    candidate_index = int(np.searchsorted(sorted_timestamps, target_timestamp))
    if candidate_index <= 0:
        return 0
    if candidate_index >= sorted_timestamps.shape[0]:
        return int(sorted_timestamps.shape[0] - 1)
    left_index = candidate_index - 1
    right_index = candidate_index
    left_error = abs(float(sorted_timestamps[left_index] - target_timestamp))
    right_error = abs(float(sorted_timestamps[right_index] - target_timestamp))
    return left_index if left_error <= right_error else right_index


def build_sync_pairs(
    d435_timestamps: np.ndarray,
    top_timestamps: np.ndarray,
    max_time_error_s: float,
) -> list[SyncPair]:
    pairs: list[SyncPair] = []
    for d435_index, d435_timestamp in enumerate(d435_timestamps):
        top_index = find_nearest_index(top_timestamps, float(d435_timestamp))
        top_timestamp = float(top_timestamps[top_index])
        delta = float(top_timestamp - d435_timestamp)
        pairs.append(
            SyncPair(
                index_d435=int(d435_index),
                index_top=int(top_index),
                timestamp_d435=float(d435_timestamp),
                timestamp_top=top_timestamp,
                delta_top_to_d435_s=delta,
                within_threshold=abs(delta) <= max_time_error_s,
            )
        )
    return pairs


def write_sync_map_jsonl(sync_pairs: list[SyncPair], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in sync_pairs:
            row = {
                "idx_d435": pair.index_d435,
                "idx_top": pair.index_top,
                "t_ref": pair.timestamp_d435,
                "t_top": pair.timestamp_top,
                "dt_top_to_ref": pair.delta_top_to_d435_s,
                "within_threshold": pair.within_threshold,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sync_summary_json(sync_pairs: list[SyncPair], output_path: Path):
    deltas = np.asarray([pair.delta_top_to_d435_s for pair in sync_pairs], dtype=float)
    within = np.asarray([pair.within_threshold for pair in sync_pairs], dtype=bool)
    summary = {
        "num_pairs": int(len(sync_pairs)),
        "num_within_threshold": int(np.sum(within)),
        "coverage_ratio": float(np.mean(within)) if len(sync_pairs) > 0 else 0.0,
        "delta_mean_s": float(np.mean(deltas)) if len(deltas) > 0 else 0.0,
        "delta_abs_p95_s": float(np.percentile(np.abs(deltas), 95)) if len(deltas) > 0 else 0.0,
        "delta_abs_max_s": float(np.max(np.abs(deltas))) if len(deltas) > 0 else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D435-to-top-camera timestamp sync map.")
    parser.add_argument(
        "--d435-episode-dir",
        required=True,
        help="Path to D435 episode directory that contains timestamps.npy",
    )
    parser.add_argument(
        "--top-timestamps",
        default=None,
        help="Path to top camera timestamps .npy (preferred)",
    )
    parser.add_argument(
        "--top-frame-meta",
        default=None,
        help="Path to top camera frame_meta.jsonl with timestamp_s field",
    )
    parser.add_argument(
        "--max-time-error-s",
        type=float,
        default=0.03,
        help="Threshold for marking sync pair as valid",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Output sync_map.jsonl path (default: <episode>/sync_map.jsonl)",
    )
    parser.add_argument(
        "--output-summary-json",
        default=None,
        help="Optional summary JSON path (default: <episode>/sync_summary.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    d435_episode_dir = Path(args.d435_episode_dir).expanduser().resolve()
    d435_timestamps_path = d435_episode_dir / "timestamps.npy"
    if not d435_timestamps_path.exists():
        raise FileNotFoundError(f"Missing D435 timestamps: {d435_timestamps_path}")

    top_timestamps_path = Path(args.top_timestamps).expanduser().resolve() if args.top_timestamps else None
    top_frame_meta_path = Path(args.top_frame_meta).expanduser().resolve() if args.top_frame_meta else None

    d435_timestamps = load_npy_timestamps(d435_timestamps_path)
    top_timestamps = load_top_timestamps(top_timestamps_path, top_frame_meta_path)
    sync_pairs = build_sync_pairs(
        d435_timestamps=d435_timestamps,
        top_timestamps=top_timestamps,
        max_time_error_s=float(args.max_time_error_s),
    )

    output_jsonl = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else d435_episode_dir / "sync_map.jsonl"
    )
    output_summary_json = (
        Path(args.output_summary_json).expanduser().resolve()
        if args.output_summary_json
        else d435_episode_dir / "sync_summary.json"
    )

    write_sync_map_jsonl(sync_pairs, output_jsonl)
    write_sync_summary_json(sync_pairs, output_summary_json)

    deltas = np.asarray([pair.delta_top_to_d435_s for pair in sync_pairs], dtype=float)
    within = np.asarray([pair.within_threshold for pair in sync_pairs], dtype=bool)
    print(f"[INFO] D435 frames: {len(d435_timestamps)}")
    print(f"[INFO] Top frames: {len(top_timestamps)}")
    print(f"[INFO] Sync pairs: {len(sync_pairs)}")
    print(f"[INFO] Within threshold: {int(np.sum(within))}/{len(sync_pairs)}")
    print(f"[INFO] Mean dt (top-d435): {float(np.mean(deltas)):.6f} s")
    print(f"[INFO] P95 |dt|: {float(np.percentile(np.abs(deltas), 95)):.6f} s")
    print(f"[INFO] Wrote: {output_jsonl}")
    print(f"[INFO] Wrote: {output_summary_json}")


if __name__ == "__main__":
    main()

