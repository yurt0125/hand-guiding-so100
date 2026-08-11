#!/usr/bin/env python3
"""Build a local LeRobot v3 dataset from exported VLA table (CSV/NPZ)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_lerobot_import(project_root: Path) -> None:
    try:
        import lerobot  # noqa: F401
        return
    except ImportError:
        import sys

        local_lerobot_src = project_root / "third-party" / "lerobot" / "src"
        if local_lerobot_src.exists():
            sys.path.insert(0, str(local_lerobot_src))
        import lerobot  # noqa: F401


def ensure_cv2_import():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Visual mode requires OpenCV (cv2). Please install opencv-python in your active environment."
        ) from exc
    return cv2


def load_vla_table(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".npz":
        npz_payload = np.load(input_path, allow_pickle=False)
        columns = [str(column_name) for column_name in npz_payload["columns"].tolist()]
        payload: dict[str, np.ndarray] = {}
        for column_name in columns:
            if column_name in npz_payload:
                payload[column_name] = npz_payload[column_name]
        return pd.DataFrame(payload)
    return pd.read_csv(input_path)


def parse_state_keys(state_keys_argument: str) -> list[str]:
    state_keys = [state_key.strip() for state_key in state_keys_argument.split(",") if state_key.strip()]
    if not state_keys:
        raise ValueError("state_keys cannot be empty")
    return state_keys


def pick_series_value(dataframe: pd.DataFrame, row_index: int, candidates: list[str], default_value: float) -> float:
    for candidate in candidates:
        if candidate not in dataframe.columns:
            continue
        value = dataframe.iat[row_index, dataframe.columns.get_loc(candidate)]
        if pd.isna(value):
            continue
        return float(value)
    return float(default_value)


def build_observation_state_vector(dataframe: pd.DataFrame, row_index: int, state_keys: list[str]) -> np.ndarray:
    observation_values: list[float] = []
    for state_key in state_keys:
        obs_column_name = f"obs_{state_key}"
        if obs_column_name not in dataframe.columns:
            raise KeyError(f"Missing observation column: {obs_column_name}")
        observation_values.append(float(dataframe.iat[row_index, dataframe.columns.get_loc(obs_column_name)]))
    return np.asarray(observation_values, dtype=np.float32)


def build_action_vector(dataframe: pd.DataFrame, row_index: int, state_keys: list[str]) -> np.ndarray:
    action_values: list[float] = []
    for state_key in state_keys:
        action_column_name = f"action_rel_{state_key}"
        if action_column_name not in dataframe.columns:
            fallback_column_name = f"action_rel_step1_{state_key}"
            if fallback_column_name not in dataframe.columns:
                raise KeyError(f"Missing action column: {action_column_name} (and fallback {fallback_column_name})")
            action_column_name = fallback_column_name
        action_value = dataframe.iat[row_index, dataframe.columns.get_loc(action_column_name)]
        action_values.append(float(0.0 if pd.isna(action_value) else action_value))
    return np.asarray(action_values, dtype=np.float32)


def resolve_episode_column(dataframe: pd.DataFrame) -> str | None:
    for candidate in ["obs_episode_id", "episode_id", "obs_episode_index", "episode_index"]:
        if candidate in dataframe.columns:
            return candidate
    return None


def build_lerobot_features(state_keys: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_keys),),
            "names": {"axes": state_keys},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(state_keys),),
            "names": {"axes": state_keys},
        },
        "observation.latency_ms": {
            "dtype": "float32",
            "shape": (1,),
            "names": {"axes": ["latency_ms"]},
        },
        "observation.sample_weight": {
            "dtype": "float32",
            "shape": (1,),
            "names": {"axes": ["sample_weight"]},
        },
        "observation.anchor_mask": {
            "dtype": "float32",
            "shape": (1,),
            "names": {"axes": ["anchor_mask"]},
        },
    }


def infer_first_valid_image_shape(
    dataframe: pd.DataFrame,
    enable_visual: bool,
    image_column: str,
    video_path: Path | None,
) -> tuple[int, int, int] | None:
    if not enable_visual:
        return None
    cv2 = ensure_cv2_import()

    if image_column in dataframe.columns:
        for image_path_value in dataframe[image_column].tolist():
            image_path = Path(str(image_path_value))
            if not image_path.exists():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width, channels = image.shape
            return int(height), int(width), int(channels)

    if video_path is not None and video_path.exists() and "obs_frame_index" in dataframe.columns:
        capture = cv2.VideoCapture(str(video_path))
        try:
            for frame_index_value in dataframe["obs_frame_index"].tolist():
                frame_index = int(frame_index_value)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                height, width, channels = frame.shape
                return int(height), int(width), int(channels)
        finally:
            capture.release()
    return None


def load_visual_frame(
    dataframe: pd.DataFrame,
    row_index: int,
    image_column: str,
    video_capture: cv2.VideoCapture | None,
) -> np.ndarray | None:
    cv2 = ensure_cv2_import()
    if image_column in dataframe.columns:
        image_path = Path(str(dataframe.iat[row_index, dataframe.columns.get_loc(image_column)]))
        if image_path.exists():
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is not None:
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if video_capture is not None and "obs_frame_index" in dataframe.columns:
        frame_index_value = dataframe.iat[row_index, dataframe.columns.get_loc("obs_frame_index")]
        frame_index = int(frame_index_value)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = video_capture.read()
        if ok and frame_bgr is not None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VLA table into local LeRobot v3 dataset.")
    parser.add_argument("--input-path", required=True, help="Input VLA table path (.csv or .npz)")
    parser.add_argument("--repo-id", required=True, help="Local LeRobot dataset repo id, e.g. rita/so100_hand_vla")
    parser.add_argument("--root", required=True, help="Root directory storing local LeRobot datasets")
    parser.add_argument("--fps", type=int, default=30, help="Dataset fps for timestamp alignment")
    parser.add_argument("--task-text", default="hand guiding imitation", help="Task text saved in each frame")
    parser.add_argument(
        "--state-keys",
        default="mid_base_x,mid_base_y,mid_base_z,tool_axis_x,tool_axis_y,tool_axis_z",
        help="Comma-separated state keys used by observation/action vectors",
    )
    parser.add_argument("--force-overwrite", action="store_true", help="Delete existing local dataset directory")
    parser.add_argument("--enable-visual", action="store_true", help="Enable visual feature observation.images.main")
    parser.add_argument("--image-column", default="obs_frame_path", help="Image path column in VLA table")
    parser.add_argument("--video-path", default=None, help="Fallback local video path used with obs_frame_index")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    ensure_lerobot_import(project_root)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    input_path = Path(args.input_path).expanduser().resolve()
    root_path = Path(args.root).expanduser().resolve()
    dataset_path = root_path / args.repo_id
    state_keys = parse_state_keys(args.state_keys)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    dataframe = load_vla_table(input_path)
    if dataframe.empty:
        raise ValueError("Input VLA table is empty")
    if "timestamp_s" not in dataframe.columns:
        raise KeyError("Input VLA table must contain timestamp_s")

    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else None
    image_shape = infer_first_valid_image_shape(
        dataframe=dataframe,
        enable_visual=bool(args.enable_visual),
        image_column=str(args.image_column),
        video_path=video_path,
    )

    if dataset_path.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"Dataset path exists: {dataset_path}. Use --force-overwrite to replace.")
        import shutil

        shutil.rmtree(dataset_path)

    features = build_lerobot_features(state_keys)
    if args.enable_visual:
        if image_shape is None:
            raise ValueError(
                "Visual mode enabled, but no readable image source found. "
                "Provide local image paths in image column or pass --video-path with obs_frame_index."
            )
        features["observation.images.main"] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root_path,
        fps=int(args.fps),
        features=features,
        robot_type="so100_5dof",
        use_videos=False,
    )

    episode_column = resolve_episode_column(dataframe)
    if episode_column is None:
        dataframe = dataframe.copy()
        dataframe["__episode_id__"] = "episode_000000"
        episode_column = "__episode_id__"

    dataframe = dataframe.reset_index(drop=True)
    video_capture = None
    if args.enable_visual and video_path is not None and video_path.exists():
        cv2 = ensure_cv2_import()
        video_capture = cv2.VideoCapture(str(video_path))

    for _, episode_dataframe in dataframe.groupby(episode_column, sort=False):
        episode_dataframe = episode_dataframe.reset_index(drop=True)
        for row_index in range(len(episode_dataframe)):
            timestamp_s = float(episode_dataframe.iat[row_index, episode_dataframe.columns.get_loc("timestamp_s")])
            frame = {
                "task": str(args.task_text),
                "timestamp": timestamp_s,
                "observation.state": build_observation_state_vector(episode_dataframe, row_index, state_keys),
                "action": build_action_vector(episode_dataframe, row_index, state_keys),
                "observation.latency_ms": np.asarray(
                    [pick_series_value(episode_dataframe, row_index, ["latency_ms"], 0.0)],
                    dtype=np.float32,
                ),
                "observation.sample_weight": np.asarray(
                    [pick_series_value(episode_dataframe, row_index, ["sample_weight"], 1.0)],
                    dtype=np.float32,
                ),
                "observation.anchor_mask": np.asarray(
                    [pick_series_value(episode_dataframe, row_index, ["anchor_mask"], 0.0)],
                    dtype=np.float32,
                ),
            }
            if args.enable_visual:
                image_array = load_visual_frame(
                    dataframe=episode_dataframe,
                    row_index=row_index,
                    image_column=str(args.image_column),
                    video_capture=video_capture,
                )
                if image_array is None:
                    raise ValueError(
                        f"Cannot load visual frame for row {row_index}. "
                        "Check image paths or provide --video-path."
                    )
                frame["observation.images.main"] = image_array
            dataset.add_frame(frame)
        dataset.save_episode()

    if video_capture is not None:
        video_capture.release()

    print(f"[INFO] Built local LeRobot dataset at: {dataset_path}")
    print(f"[INFO] repo_id={args.repo_id}")
    print(f"[INFO] fps={args.fps}")
    print(f"[INFO] state_keys={state_keys}")
    print(f"[INFO] visual_enabled={bool(args.enable_visual)}")
    print("[INFO] Suggested ACT train command:")
    print(
        "PYTHONPATH=third-party/lerobot/src lerobot-train "
        f"--policy.type=act --dataset.repo_id={args.repo_id} --dataset.root={root_path} "
        "--dataset.use_imagenet_stats=false "
        "--policy.device=cuda "
        "--output_dir=outputs/act_so100"
    )


if __name__ == "__main__":
    main()
