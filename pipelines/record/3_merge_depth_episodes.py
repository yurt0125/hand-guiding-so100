#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm as _real_tqdm
except Exception:
    _real_tqdm = None


def _tqdm(iterable, **kwargs):
    if _real_tqdm is None:
        return iterable
    return _real_tqdm(iterable, **kwargs)


def merge_depth_episodes(dataset_root: Path, camera_key: str = "d435", compressed: bool = True) -> None:
    camera_root = dataset_root / "depth" / camera_key
    episodes_root = camera_root / "episodes"
    if not episodes_root.exists():
        raise FileNotFoundError(
            f"Episodes directory not found: {episodes_root}\n"
            "This dataset has no exported depth episodes. Re-record with "
            "pipelines/record/3_rawdata_record.py and a RealSense camera configured with "
            "'use_depth': true. If you intentionally disabled depth export, remove "
            "--no-export-depth-npz."
        )

    episode_paths = sorted(episodes_root.glob("episode-*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode npz found under: {episodes_root}")

    index_rows: list[dict[str, int]] = []
    manifest_rows: list[dict[str, object]] = []
    episode_meta: list[tuple[int, Path, int]] = []

    cursor = 0
    for ep_path in _tqdm(episode_paths, desc="scan episodes", unit="ep"):
        episode_name = ep_path.stem
        episode_index = int(episode_name.split("-")[-1])
        depth = np.load(ep_path)["depth_mm"]
        if depth.ndim != 3:
            raise ValueError(f"{ep_path} depth_mm must be 3D [T,H,W], got {depth.shape}")

        length = int(depth.shape[0])
        start = cursor
        end = cursor + length
        cursor = end

        episode_meta.append((episode_index, ep_path, length))
        index_rows.append({"episode_index": episode_index, "start": start, "end": end})
        manifest_rows.append(
            {
                "episode_index": episode_index,
                "path": str(ep_path.relative_to(dataset_root)),
                "frames": length,
            }
        )

    if not episode_meta:
        raise RuntimeError("No valid episode depths to merge.")

    # Build merged stack with low memory overhead via memmap.
    sample = np.load(episode_meta[0][1])["depth_mm"]
    h, w = int(sample.shape[1]), int(sample.shape[2])
    total_frames = sum(length for _, _, length in episode_meta)
    tmp_path = Path(tempfile.mkstemp(prefix="depth_merge_", suffix=".mmap")[1])
    try:
        merged_mm = np.memmap(tmp_path, mode="w+", dtype=np.uint16, shape=(total_frames, h, w))
        cur = 0
        for _, ep_path, length in _tqdm(episode_meta, desc="merge depth", unit="ep"):
            arr = np.load(ep_path)["depth_mm"].astype(np.uint16, copy=False)
            merged_mm[cur : cur + length] = arr
            cur += length
        merged_mm.flush()

        if compressed:
            np.savez_compressed(camera_root / "depth.npz", depth_mm=merged_mm)
        else:
            np.savez(camera_root / "depth.npz", depth_mm=merged_mm)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    (camera_root / "depth_index.json").write_text(
        json.dumps(
            {
                "camera_key": camera_key,
                "total_frames": int(total_frames),
                "episodes": index_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (episodes_root / "manifest.json").write_text(
        json.dumps(
            {
                "camera_key": camera_key,
                "episodes": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Keep one canonical intrinsics.json at camera root if present in episodes folder style datasets.
    # If it already exists, keep it unchanged.
    if not (camera_root / "intrinsics.json").exists():
        candidates = sorted(episodes_root.glob("episode-*.intrinsics.json"))
        if candidates:
            (camera_root / "intrinsics.json").write_text(candidates[0].read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[INFO] merged episodes: {len(episode_paths)}")
    print(f"[INFO] total frames: {total_frames}")
    print(f"[INFO] wrote: {camera_root / 'depth.npz'}")
    print(f"[INFO] wrote: {camera_root / 'depth_index.json'}")
    print(f"[INFO] wrote: {episodes_root / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually merge episode depth npz files into one depth.npz.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root, e.g. data/raw/5.8_act_test2")
    parser.add_argument("--camera-key", default="d435", help="Camera key under depth/, default: d435")
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Write uncompressed depth.npz for faster output (default is compressed).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    merge_depth_episodes(dataset_root=dataset_root, camera_key=args.camera_key, compressed=not args.uncompressed)


if __name__ == "__main__":
    main()
