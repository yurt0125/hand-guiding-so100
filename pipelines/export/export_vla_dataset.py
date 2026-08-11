#!/usr/bin/env python3
"""Export trajectory CSV into VLA-friendly observation/action-chunk dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_STATE_COLUMNS = [
    "mid_base_x",
    "mid_base_y",
    "mid_base_z",
    "tool_axis_x",
    "tool_axis_y",
    "tool_axis_z",
]


def _read_optional_float(dataframe: pd.DataFrame, row_index: int, column_candidates: list[str]) -> float | None:
    for column_name in column_candidates:
        if column_name not in dataframe.columns:
            continue
        value = dataframe.iat[row_index, dataframe.columns.get_loc(column_name)]
        if pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _read_optional_bool(dataframe: pd.DataFrame, row_index: int, column_candidates: list[str]) -> bool | None:
    for column_name in column_candidates:
        if column_name not in dataframe.columns:
            continue
        value = dataframe.iat[row_index, dataframe.columns.get_loc(column_name)]
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer, float, np.floating)):
            return bool(float(value) > 0.0)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes"}:
            return True
        if text in {"0", "false", "no"}:
            return False
    return None


def compute_anchor_mask(dataframe: pd.DataFrame, row_index: int) -> int:
    fused_flag = _read_optional_bool(
        dataframe,
        row_index,
        ["grasp_fused", "obs_grasp_fused"],
    )
    if fused_flag is True:
        return 1
    fused_alpha = _read_optional_float(
        dataframe,
        row_index,
        ["grasp_alpha", "obs_grasp_alpha"],
    )
    if fused_alpha is not None and fused_alpha > 0.0:
        return 1
    return 0


def compute_latency_ms(dataframe: pd.DataFrame, row_index: int, fallback_delta_t_s: float) -> float:
    explicit_latency_ms = _read_optional_float(
        dataframe,
        row_index,
        ["latency_ms", "obs_latency_ms"],
    )
    if explicit_latency_ms is not None:
        return explicit_latency_ms
    return float(max(0.0, fallback_delta_t_s) * 1000.0)


def compute_sample_weight(dataframe: pd.DataFrame, row_index: int, anchor_mask: int) -> float:
    mapping_quality_score = _read_optional_float(
        dataframe,
        row_index,
        [
            "mapping_quality_score",
            "mapping_quality.score",
            "runtime.mapping_quality.score",
            "obs_mapping_quality_score",
        ],
    )
    if mapping_quality_score is None:
        mapping_quality_score = 1.0
    mapping_quality_score = float(np.clip(mapping_quality_score, 0.0, 1.0))

    pinch_confidence = _read_optional_float(dataframe, row_index, ["min_pinch_conf", "obs_min_pinch_conf"])
    if pinch_confidence is None:
        kp02_conf = _read_optional_float(dataframe, row_index, ["kp02_conf", "obs_kp02_conf"])
        kp05_conf = _read_optional_float(dataframe, row_index, ["kp05_conf", "obs_kp05_conf"])
        if kp02_conf is not None and kp05_conf is not None:
            pinch_confidence = min(kp02_conf, kp05_conf)
    if pinch_confidence is None:
        pinch_confidence = 1.0
    pinch_confidence = float(np.clip(pinch_confidence, 0.0, 1.0))

    grasp_score = _read_optional_float(dataframe, row_index, ["grasp_score", "obs_grasp_score"])
    if grasp_score is None:
        grasp_score = 0.0
    grasp_confidence = float(np.clip(grasp_score / 2.0, 0.0, 1.0))

    if anchor_mask == 1:
        total_weight = 0.50 * mapping_quality_score + 0.20 * pinch_confidence + 0.30 * grasp_confidence
    else:
        total_weight = 0.70 * mapping_quality_score + 0.30 * pinch_confidence
    return float(np.clip(total_weight, 0.0, 1.0))


def parse_column_list(argument_value: str | None, default_columns: Iterable[str]) -> list[str]:
    if argument_value is None or not argument_value.strip():
        return list(default_columns)
    return [column.strip() for column in argument_value.split(",") if column.strip()]


def ensure_timestamp_seconds(dataframe: pd.DataFrame, fallback_fps: float) -> pd.DataFrame:
    dataframe = dataframe.copy()
    if "timestamp_s" in dataframe.columns:
        dataframe["timestamp_s"] = dataframe["timestamp_s"].astype(float)
        return dataframe
    if "t" in dataframe.columns:
        dataframe["timestamp_s"] = dataframe["t"].astype(float)
        return dataframe
    if "frame_index" in dataframe.columns:
        if fallback_fps <= 0:
            raise ValueError("--fps must be > 0 when frame_index fallback is used")
        dataframe["timestamp_s"] = dataframe["frame_index"].astype(float) / float(fallback_fps)
        return dataframe
    if fallback_fps <= 0:
        raise ValueError("--fps must be > 0 when synthetic timestamp fallback is used")
    dataframe["timestamp_s"] = np.arange(len(dataframe), dtype=float) / float(fallback_fps)
    return dataframe


def build_training_rows(
    dataframe: pd.DataFrame,
    state_columns: list[str],
    chunk_size: int,
    drop_incomplete_chunks: bool,
) -> pd.DataFrame:
    dataframe = dataframe.reset_index(drop=False).rename(columns={"index": "source_row_index"})
    state_matrix = dataframe[state_columns].to_numpy(dtype=float)
    timestamps = dataframe["timestamp_s"].to_numpy(dtype=float)

    output_rows: list[dict[str, float | int | str | bool]] = []
    total_rows = len(dataframe)

    for row_index in range(total_rows):
        if row_index + 1 >= total_rows:
            break

        if drop_incomplete_chunks and (row_index + chunk_size >= total_rows):
            break

        row_payload: dict[str, float | int | str | bool] = {}
        observation_row = dataframe.iloc[row_index]
        for column_name, value in observation_row.items():
            row_payload[f"obs_{column_name}"] = value

        row_payload["sample_index"] = row_index
        row_payload["timestamp_s"] = float(timestamps[row_index])
        row_payload["timestamp_next_s"] = float(timestamps[row_index + 1])
        row_payload["delta_t_s"] = float(timestamps[row_index + 1] - timestamps[row_index])
        row_payload["anchor_mask"] = int(compute_anchor_mask(dataframe, row_index))
        row_payload["latency_ms"] = float(
            compute_latency_ms(
                dataframe=dataframe,
                row_index=row_index,
                fallback_delta_t_s=row_payload["delta_t_s"],
            )
        )
        row_payload["sample_weight"] = float(
            compute_sample_weight(
                dataframe=dataframe,
                row_index=row_index,
                anchor_mask=row_payload["anchor_mask"],
            )
        )

        current_state = state_matrix[row_index]
        next_state = state_matrix[row_index + 1]
        single_step_action = next_state - current_state
        for state_column_index, state_column_name in enumerate(state_columns):
            row_payload[f"action_rel_{state_column_name}"] = float(single_step_action[state_column_index])

        for chunk_step in range(1, chunk_size + 1):
            target_index = row_index + chunk_step
            if target_index < total_rows:
                target_state = state_matrix[target_index]
                action_delta = target_state - current_state
                row_payload[f"action_valid_step{chunk_step}"] = True
                row_payload[f"chunk_delta_t_step{chunk_step}_s"] = float(
                    timestamps[target_index] - timestamps[row_index]
                )
                for state_column_index, state_column_name in enumerate(state_columns):
                    row_payload[f"action_rel_step{chunk_step}_{state_column_name}"] = float(
                        action_delta[state_column_index]
                    )
            else:
                row_payload[f"action_valid_step{chunk_step}"] = False
                row_payload[f"chunk_delta_t_step{chunk_step}_s"] = np.nan
                for state_column_name in state_columns:
                    row_payload[f"action_rel_step{chunk_step}_{state_column_name}"] = np.nan

        output_rows.append(row_payload)

    return pd.DataFrame(output_rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trajectory CSV to VLA-ready training table")
    parser.add_argument("--input-csv", required=True, help="Input trajectory CSV")
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output path (.csv or .npz). Kept for backward compatibility.",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "csv", "npz"],
        default="auto",
        help="Export format. 'auto' infers from output suffix.",
    )
    parser.add_argument(
        "--state-columns",
        default=",".join(DEFAULT_STATE_COLUMNS),
        help="Comma-separated state columns used to compute relative actions",
    )
    parser.add_argument("--chunk-size", type=int, default=10, help="Future action chunk horizon")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback fps for timestamp construction")
    parser.add_argument(
        "--keep-hand-found-only",
        action="store_true",
        help="Use only rows where hand_found == 1 when that column exists",
    )
    parser.add_argument(
        "--drop-incomplete-chunks",
        action="store_true",
        help="Drop tail samples without full future chunk",
    )
    return parser.parse_args()


def resolve_output_format(output_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    return "npz" if output_path.suffix.lower() == ".npz" else "csv"


def dataframe_to_npz_payload(dataframe: pd.DataFrame) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    payload["columns"] = np.asarray(list(dataframe.columns), dtype=str)
    payload["row_count"] = np.asarray([len(dataframe)], dtype=np.int64)
    payload["schema_json"] = np.asarray(
        [json.dumps({column: str(dtype) for column, dtype in dataframe.dtypes.items()}, ensure_ascii=False)],
        dtype=str,
    )

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        if pd.api.types.is_bool_dtype(series):
            payload[column_name] = series.astype(np.uint8).to_numpy()
        elif pd.api.types.is_integer_dtype(series):
            payload[column_name] = series.astype(np.int64).to_numpy()
        elif pd.api.types.is_float_dtype(series):
            if "timestamp" in column_name or "delta_t" in column_name:
                payload[column_name] = series.astype(np.float64).to_numpy()
            else:
                payload[column_name] = series.astype(np.float32).to_numpy()
        else:
            payload[column_name] = series.astype(str).to_numpy(dtype=str)
    return payload


def save_exported_dataset(exported_dataframe: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        exported_dataframe.to_csv(output_path, index=False)
        return
    payload = dataframe_to_npz_payload(exported_dataframe)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    arguments = parse_arguments()
    input_path = Path(arguments.input_csv).expanduser().resolve()
    output_path = Path(arguments.output_csv).expanduser().resolve()
    output_format = resolve_output_format(output_path=output_path, requested_format=arguments.output_format)
    state_columns = parse_column_list(arguments.state_columns, DEFAULT_STATE_COLUMNS)

    if arguments.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    if arguments.keep_hand_found_only and "hand_found" in dataframe.columns:
        dataframe = dataframe[dataframe["hand_found"] == 1].copy()
    if dataframe.empty:
        raise ValueError("No rows available after filtering")

    missing_columns = [column_name for column_name in state_columns if column_name not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing state columns: {missing_columns}")

    dataframe = dataframe.dropna(subset=state_columns).copy()
    if len(dataframe) < 2:
        raise ValueError("Need at least 2 valid rows to compute relative actions")

    dataframe = ensure_timestamp_seconds(dataframe, fallback_fps=arguments.fps)
    dataframe["t"] = dataframe["timestamp_s"] - float(dataframe["timestamp_s"].iloc[0])

    exported_dataframe = build_training_rows(
        dataframe=dataframe,
        state_columns=state_columns,
        chunk_size=arguments.chunk_size,
        drop_incomplete_chunks=arguments.drop_incomplete_chunks,
    )
    if exported_dataframe.empty:
        raise ValueError("No training rows produced. Check chunk-size and input length.")

    save_exported_dataset(exported_dataframe=exported_dataframe, output_path=output_path, output_format=output_format)

    print(f"[INFO] Input rows: {len(dataframe)}")
    print(f"[INFO] Output rows: {len(exported_dataframe)}")
    print(f"[INFO] State columns: {state_columns}")
    print(f"[INFO] Chunk size: {arguments.chunk_size}")
    print(f"[INFO] Output format: {output_format}")
    print(f"[INFO] Output path: {output_path}")


if __name__ == "__main__":
    main()
