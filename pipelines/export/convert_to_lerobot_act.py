#!/usr/bin/env python3
"""Convert exported VLA table (CSV/NPZ) into an ACT-friendly episode dataset package.

This script does not depend on a specific trainer implementation. It creates a
clean, explicit dataset package that can be consumed by LeRobot/ACT training code:

dataset_root/
  metadata.json
  episodes/
    episode_000000.npz
  splits/
    train_episode_ids.npy
    val_episode_ids.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STATE_KEYS = [
    "mid_base_x",
    "mid_base_y",
    "mid_base_z",
    "tool_axis_x",
    "tool_axis_y",
    "tool_axis_z",
]


def parse_state_keys(state_keys_argument: str | None) -> list[str]:
    if state_keys_argument is None or not state_keys_argument.strip():
        return list(DEFAULT_STATE_KEYS)
    return [state_key.strip() for state_key in state_keys_argument.split(",") if state_key.strip()]


def load_vla_table(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".npz":
        npz_payload = np.load(input_path, allow_pickle=False)
        if "columns" not in npz_payload:
            raise KeyError("NPZ payload missing 'columns'.")
        columns = [str(column_name) for column_name in npz_payload["columns"].tolist()]
        reconstructed: dict[str, np.ndarray] = {}
        for column_name in columns:
            if column_name in npz_payload:
                reconstructed[column_name] = npz_payload[column_name]
        return pd.DataFrame(reconstructed)
    return pd.read_csv(input_path)


def resolve_episode_ids(dataframe: pd.DataFrame) -> np.ndarray:
    candidate_columns = [
        "obs_episode_id",
        "episode_id",
        "obs_episode_index",
        "episode_index",
    ]
    for candidate_column in candidate_columns:
        if candidate_column in dataframe.columns:
            return dataframe[candidate_column].astype(str).to_numpy()
    return np.asarray(["episode_000000"] * len(dataframe), dtype=str)


def resolve_image_reference_column(dataframe: pd.DataFrame) -> str | None:
    for candidate_column in ["obs_frame_path", "obs_image_path", "frame_path", "image_path"]:
        if candidate_column in dataframe.columns:
            return candidate_column
    return None


def collect_action_chunk(
    dataframe: pd.DataFrame,
    state_keys: list[str],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = len(dataframe)
    state_dim = len(state_keys)
    action_chunk = np.zeros((row_count, chunk_size, state_dim), dtype=np.float32)
    action_valid_mask = np.zeros((row_count, chunk_size), dtype=np.bool_)

    for chunk_step in range(1, chunk_size + 1):
        valid_column = f"action_valid_step{chunk_step}"
        if valid_column in dataframe.columns:
            valid_mask = dataframe[valid_column].astype(bool).to_numpy()
        else:
            valid_mask = np.ones(row_count, dtype=bool)
        action_valid_mask[:, chunk_step - 1] = valid_mask

        for state_index, state_key in enumerate(state_keys):
            action_column = f"action_rel_step{chunk_step}_{state_key}"
            if action_column not in dataframe.columns:
                raise KeyError(f"Missing action column: {action_column}")
            action_values = pd.to_numeric(dataframe[action_column], errors="coerce").to_numpy(dtype=np.float32)
            nan_count = int(np.isnan(action_values).sum())
            if nan_count > 0:
                import logging
                logging.warning(
                    "convert_to_lerobot_act: %d NaN values in column '%s' replaced with 0.0",
                    nan_count, action_column,
                )
            action_chunk[:, chunk_step - 1, state_index] = np.nan_to_num(action_values, nan=0.0)

    return action_chunk, action_valid_mask


def collect_observation_state(dataframe: pd.DataFrame, state_keys: list[str]) -> np.ndarray:
    observation_columns = []
    for state_key in state_keys:
        candidate_column = f"obs_{state_key}"
        if candidate_column not in dataframe.columns:
            raise KeyError(f"Missing observation state column: {candidate_column}")
        observation_columns.append(candidate_column)
    return dataframe[observation_columns].to_numpy(dtype=np.float32)


def write_episode_npz(
    episode_output_path: Path,
    episode_dataframe: pd.DataFrame,
    state_keys: list[str],
    chunk_size: int,
    image_reference_column: str | None,
) -> dict[str, Any]:
    observation_state = collect_observation_state(episode_dataframe, state_keys=state_keys)
    action_chunk, action_valid_mask = collect_action_chunk(
        dataframe=episode_dataframe,
        state_keys=state_keys,
        chunk_size=chunk_size,
    )

    timestamp_seconds = pd.to_numeric(
        episode_dataframe["timestamp_s"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    latency_milliseconds = pd.to_numeric(
        episode_dataframe["latency_ms"] if "latency_ms" in episode_dataframe.columns else 0.0,
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    sample_weight = pd.to_numeric(
        episode_dataframe["sample_weight"] if "sample_weight" in episode_dataframe.columns else 1.0,
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    anchor_mask = (
        episode_dataframe["anchor_mask"].astype(np.int64).to_numpy()
        if "anchor_mask" in episode_dataframe.columns
        else np.zeros(len(episode_dataframe), dtype=np.int64)
    )

    payload: dict[str, np.ndarray] = {
        "timestamp_s": np.nan_to_num(timestamp_seconds, nan=0.0),
        "observation_state": np.nan_to_num(observation_state, nan=0.0),
        "action_chunk": np.nan_to_num(action_chunk, nan=0.0),
        "action_valid_mask": action_valid_mask,
        "latency_ms": np.nan_to_num(latency_milliseconds, nan=0.0),
        "sample_weight": np.nan_to_num(sample_weight, nan=1.0),
        "anchor_mask": anchor_mask,
    }

    if image_reference_column is not None:
        image_references = episode_dataframe[image_reference_column].astype(str).to_numpy(dtype=str)
        payload["observation_image_reference"] = image_references

    episode_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(episode_output_path, **payload)

    return {
        "rows": int(len(episode_dataframe)),
        "has_images": bool(image_reference_column is not None),
        "path": str(episode_output_path),
    }


def split_episode_ids(episode_ids: list[str], validation_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = sorted(set(episode_ids))
    if len(unique_ids) <= 1 or validation_ratio <= 0.0:
        return np.asarray(unique_ids, dtype=str), np.asarray([], dtype=str)

    rng = np.random.default_rng(seed=seed)
    shuffled_indices = np.arange(len(unique_ids))
    rng.shuffle(shuffled_indices)

    validation_count = int(np.ceil(len(unique_ids) * validation_ratio))
    validation_count = min(max(validation_count, 1), len(unique_ids) - 1)
    validation_indices = set(shuffled_indices[:validation_count].tolist())

    train_ids: list[str] = []
    validation_ids: list[str] = []
    for index, episode_id in enumerate(unique_ids):
        if index in validation_indices:
            validation_ids.append(episode_id)
        else:
            train_ids.append(episode_id)
    return np.asarray(train_ids, dtype=str), np.asarray(validation_ids, dtype=str)


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Convert VLA table to LeRobot/ACT-friendly episode package.")
    argument_parser.add_argument("--input-path", required=True, help="Input VLA table (.csv or .npz)")
    argument_parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    argument_parser.add_argument(
        "--state-keys",
        default=",".join(DEFAULT_STATE_KEYS),
        help="Comma-separated state keys used by observation/action tensors",
    )
    argument_parser.add_argument("--chunk-size", type=int, default=10, help="Chunk horizon")
    argument_parser.add_argument("--dataset-name", default="so100_hand_guiding_act")
    argument_parser.add_argument("--validation-ratio", type=float, default=0.1)
    argument_parser.add_argument("--split-seed", type=int, default=42)
    arguments = argument_parser.parse_args()

    if arguments.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if not (0.0 <= arguments.validation_ratio < 1.0):
        raise ValueError("--validation-ratio must be in [0.0, 1.0)")

    input_path = Path(arguments.input_path).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    episodes_output_dir = output_dir / "episodes"
    splits_output_dir = output_dir / "splits"
    state_keys = parse_state_keys(arguments.state_keys)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    dataframe = load_vla_table(input_path)
    if dataframe.empty:
        raise ValueError("Input VLA table is empty")
    if "timestamp_s" not in dataframe.columns:
        raise KeyError("Input VLA table must contain 'timestamp_s'")

    dataframe = dataframe.reset_index(drop=True)
    dataframe["episode_id"] = resolve_episode_ids(dataframe)
    image_reference_column = resolve_image_reference_column(dataframe)

    episode_summaries: list[dict[str, Any]] = []
    ordered_episode_ids: list[str] = []
    for episode_position, (episode_id, episode_dataframe) in enumerate(dataframe.groupby("episode_id", sort=False)):
        safe_episode_id = f"episode_{episode_position:06d}"
        ordered_episode_ids.append(safe_episode_id)
        episode_output_path = episodes_output_dir / f"{safe_episode_id}.npz"
        summary = write_episode_npz(
            episode_output_path=episode_output_path,
            episode_dataframe=episode_dataframe.reset_index(drop=True),
            state_keys=state_keys,
            chunk_size=arguments.chunk_size,
            image_reference_column=image_reference_column,
        )
        summary["episode_id"] = safe_episode_id
        summary["source_episode_id"] = str(episode_id)
        episode_summaries.append(summary)

    train_episode_ids, validation_episode_ids = split_episode_ids(
        episode_ids=ordered_episode_ids,
        validation_ratio=float(arguments.validation_ratio),
        seed=int(arguments.split_seed),
    )

    splits_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(splits_output_dir / "train_episode_ids.npy", train_episode_ids)
    np.save(splits_output_dir / "val_episode_ids.npy", validation_episode_ids)

    metadata = {
        "dataset_name": str(arguments.dataset_name),
        "source_input_path": str(input_path),
        "state_keys": state_keys,
        "chunk_size": int(arguments.chunk_size),
        "row_count": int(len(dataframe)),
        "episode_count": int(len(episode_summaries)),
        "image_reference_column": image_reference_column,
        "train_episode_count": int(len(train_episode_ids)),
        "val_episode_count": int(len(validation_episode_ids)),
        "episodes": episode_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    print(f"[INFO] Input table rows: {len(dataframe)}")
    print(f"[INFO] Output dataset dir: {output_dir}")
    print(f"[INFO] Episode count: {len(episode_summaries)}")
    print(f"[INFO] Train/Val episodes: {len(train_episode_ids)}/{len(validation_episode_ids)}")
    print(f"[INFO] State keys: {state_keys}")
    print(f"[INFO] Chunk size: {arguments.chunk_size}")


if __name__ == "__main__":
    main()
