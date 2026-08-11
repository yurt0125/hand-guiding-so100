#!/usr/bin/env python3
"""Validate VLA exported CSV quality and print a concise report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def collect_action_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if column.startswith("action_rel_")]


def collect_chunk_mask_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if column.startswith("action_valid_step")]


def validate_time_monotonic(dataframe: pd.DataFrame) -> tuple[bool, int]:
    timestamps = dataframe["timestamp_s"].to_numpy(dtype=float)
    if len(timestamps) <= 1:
        return True, 0
    non_increasing = int(np.sum(np.diff(timestamps) <= 0))
    return non_increasing == 0, non_increasing


def compute_nan_ratio(dataframe: pd.DataFrame, columns: list[str]) -> float:
    if not columns:
        return 0.0
    values = dataframe[columns].to_numpy(dtype=float)
    return float(np.isnan(values).sum() / values.size)


def compute_chunk_valid_ratio(dataframe: pd.DataFrame, chunk_mask_columns: list[str]) -> float:
    if not chunk_mask_columns:
        return 0.0
    mask_values = dataframe[chunk_mask_columns].astype(bool).to_numpy()
    return float(mask_values.mean())


def print_distribution_summary(dataframe: pd.DataFrame, action_columns: list[str], limit: int = 8) -> None:
    print("[ACTION STATS]")
    if not action_columns:
        print("- No action columns found")
        return
    displayed_columns = action_columns[:limit]
    for column in displayed_columns:
        column_values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if column_values.empty:
            print(f"- {column}: empty")
            continue
        print(
            f"- {column}: mean={column_values.mean():.6f}, std={column_values.std(ddof=0):.6f}, "
            f"p05={column_values.quantile(0.05):.6f}, p95={column_values.quantile(0.95):.6f}"
        )
    if len(action_columns) > limit:
        print(f"- ... {len(action_columns) - limit} more action columns omitted")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported VLA dataset (CSV or NPZ)")
    parser.add_argument("--input-path", required=True, help="Path to exported VLA dataset (.csv or .npz)")
    parser.add_argument(
        "--max-action-nan-ratio",
        type=float,
        default=0.10,
        help="Fail threshold for action NaN ratio",
    )
    parser.add_argument(
        "--max-non-increasing-timestamps",
        type=int,
        default=0,
        help="Fail threshold for non-increasing timestamp count",
    )
    parser.add_argument(
        "--min-chunk-valid-ratio",
        type=float,
        default=0.70,
        help="Fail threshold for chunk valid ratio",
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".npz":
        npz_payload = np.load(input_path, allow_pickle=False)
        columns = [str(column_name) for column_name in npz_payload["columns"].tolist()]
        reconstructed_data: dict[str, np.ndarray] = {}
        for column_name in columns:
            if column_name in npz_payload:
                reconstructed_data[column_name] = npz_payload[column_name]
        return pd.DataFrame(reconstructed_data)
    return pd.read_csv(input_path)


def main() -> None:
    arguments = parse_arguments()
    input_path = Path(arguments.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    dataframe = load_dataset(input_path)
    if dataframe.empty:
        raise ValueError("Input dataset is empty")

    required_columns = ["timestamp_s", "timestamp_next_s", "delta_t_s"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    action_columns = collect_action_columns(dataframe)
    chunk_mask_columns = collect_chunk_mask_columns(dataframe)

    is_time_monotonic, non_increasing_count = validate_time_monotonic(dataframe)
    action_nan_ratio = compute_nan_ratio(dataframe, action_columns)
    chunk_valid_ratio = compute_chunk_valid_ratio(dataframe, chunk_mask_columns)

    print("[VLA VALIDATION REPORT]")
    print(f"- file: {input_path}")
    print(f"- rows: {len(dataframe)}")
    print(f"- cols: {len(dataframe.columns)}")
    print(f"- action_cols: {len(action_columns)}")
    print(f"- chunk_mask_cols: {len(chunk_mask_columns)}")
    print(f"- time_monotonic: {is_time_monotonic} (non_increasing={non_increasing_count})")
    print(f"- action_nan_ratio: {action_nan_ratio:.6f}")
    print(f"- chunk_valid_ratio: {chunk_valid_ratio:.6f}")
    print_distribution_summary(dataframe, action_columns)

    failed_checks: list[str] = []
    if non_increasing_count > arguments.max_non_increasing_timestamps:
        failed_checks.append(
            f"non_increasing_timestamps={non_increasing_count} > {arguments.max_non_increasing_timestamps}"
        )
    if action_nan_ratio > arguments.max_action_nan_ratio:
        failed_checks.append(
            f"action_nan_ratio={action_nan_ratio:.6f} > {arguments.max_action_nan_ratio:.6f}"
        )
    if chunk_valid_ratio < arguments.min_chunk_valid_ratio:
        failed_checks.append(
            f"chunk_valid_ratio={chunk_valid_ratio:.6f} < {arguments.min_chunk_valid_ratio:.6f}"
        )

    if failed_checks:
        print("[RESULT] FAIL")
        for failure_item in failed_checks:
            print(f"- {failure_item}")
        raise SystemExit(2)

    print("[RESULT] PASS")


if __name__ == "__main__":
    main()
