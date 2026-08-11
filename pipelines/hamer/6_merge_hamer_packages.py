#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _iter(items, desc: str, unit: str = "item"):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit=unit)


def _load_pkl(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _merge_pkl_payloads(payloads: list[Any], source_names: list[str], merged_name: str) -> Any:
    first = payloads[0]

    if isinstance(first, list):
        merged: list[Any] = []
        for seq in _iter(payloads, "merge pkl payloads", "src"):
            if not isinstance(seq, list):
                raise TypeError(f"Mixed pkl types: expected list, got {type(seq)}")
            for rec in _iter(seq, "append frames", "frame"):
                merged.append(rec)
        return merged

    if isinstance(first, dict):
        if "results" in first and isinstance(first.get("results"), list):
            merged = dict(first)
            merged_results: list[Any] = []
            for payload in _iter(payloads, "merge pkl payloads", "src"):
                if not isinstance(payload, dict):
                    raise TypeError(f"Mixed pkl types: expected dict, got {type(payload)}")
                results = payload.get("results")
                if not isinstance(results, list):
                    raise ValueError("dict payload missing list key 'results'")
                for rec in _iter(results, "append frames", "frame"):
                    merged_results.append(rec)
            merged["results"] = merged_results
            if "total_frames" in merged:
                merged["total_frames"] = len(merged_results)
            return merged

        # Frame-map dict: {".../000001.jpg": {...}, ...}
        merged_map: dict[str, Any] = {}
        global_idx = 1
        for src_name, payload in zip(source_names, _iter(payloads, "merge pkl payloads", "src")):
            if not isinstance(payload, dict):
                raise TypeError(f"Mixed pkl types: expected dict, got {type(payload)}")

            keys = list(payload.keys())

            def _k(k: str):
                stem = Path(k).stem
                try:
                    return (0, int(stem))
                except Exception:
                    return (1, k)

            keys = sorted(keys, key=_k)
            for old_key in _iter(keys, f"append {src_name}", "frame"):
                rec = payload[old_key]
                new_key = f"{merged_name}/{global_idx:06d}.jpg"
                merged_map[new_key] = rec
                global_idx += 1
        return merged_map

    raise TypeError(f"Unsupported pkl payload type: {type(first)}")


def _concat_videos_ffmpeg(video_paths: list[Path], out_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fp:
        list_file = Path(fp.name)
        for v in video_paths:
            fp.write(f"file '{v.as_posix()}'\n")

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0 and out_path.exists()
    finally:
        list_file.unlink(missing_ok=True)


def _concat_videos_cv2(video_paths: list[Path], out_path: Path) -> dict[str, Any]:
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_cap = cv2.VideoCapture(str(video_paths[0]))
    if not first_cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_paths[0]}")
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(first_cap.get(cv2.CAP_PROP_FPS)) or 30.0
    first_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {out_path}")

    total_written = 0
    per_source = []
    try:
        for vp in _iter(video_paths, "concat video", "src"):
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {vp}")
            src_written = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                src_written += 1
                total_written += 1
            cap.release()
            per_source.append({"video": str(vp), "written_frames": src_written})
    finally:
        writer.release()

    return {
        "frames": total_written,
        "fps": fps,
        "width": width,
        "height": height,
        "per_source": per_source,
    }


def _concat_videos(video_paths: list[Path], out_path: Path) -> dict[str, Any]:
    if _concat_videos_ffmpeg(video_paths, out_path):
        return {
            "method": "ffmpeg_copy",
            "output": str(out_path),
            "sources": [str(p) for p in video_paths],
        }
    cv2_info = _concat_videos_cv2(video_paths, out_path)
    return {
        "method": "opencv_reencode",
        "output": str(out_path),
        "sources": [str(p) for p in video_paths],
        **cv2_info,
    }


def _merge_raw_packages(raw_input_dirs: list[Path], raw_output_root: Path, merged_name: str) -> None:
    merged_root = raw_output_root / merged_name
    (merged_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (merged_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (merged_root / "videos" / "observation.images.d435" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (merged_root / "depth" / "d435" / "episodes").mkdir(parents=True, exist_ok=True)

    # --- merge data parquet ---
    data_parts = []
    frame_cursor = 0
    ep_cursor = 0
    source_episode_counts = []
    source_frame_counts = []

    for raw_dir in _iter(raw_input_dirs, "merge raw data parquet", "src"):
        data_pq = raw_dir / "data" / "chunk-000" / "file-000.parquet"
        if not data_pq.exists():
            raise FileNotFoundError(f"Missing raw data parquet: {data_pq}")
        df = pd.read_parquet(data_pq)

        # Keep original semantics:
        # - episode_index: global offset after merge
        # - frame_index: per-episode local index starting from 0
        # - index: global continuous index over full dataset
        local_ep_max = int(df["episode_index"].max()) if len(df) > 0 else -1
        source_episode_counts.append(local_ep_max + 1)
        source_frame_counts.append(int(len(df)))

        df = df.copy()
        local_episode_index = df["episode_index"].astype(np.int64)
        df["frame_index"] = local_episode_index.groupby(local_episode_index, sort=False).cumcount().astype(np.int64)
        df["episode_index"] = df["episode_index"].astype(np.int64) + ep_cursor
        df["index"] = np.arange(frame_cursor, frame_cursor + len(df), dtype=np.int64)
        # keep timestamp relative to episode start, do not force global timeline
        data_parts.append(df)

        frame_cursor += len(df)
        ep_cursor += local_ep_max + 1

    merged_data = pd.concat(data_parts, axis=0, ignore_index=True)
    out_data_pq = merged_root / "data" / "chunk-000" / "file-000.parquet"
    merged_data.to_parquet(out_data_pq, index=False)

    # --- merge meta episodes parquet ---
    ep_parts = []
    global_ep_offset = 0
    global_frame_offset = 0
    global_time_offset = 0.0

    for raw_dir, src_ep_n, src_frame_n in zip(raw_input_dirs, source_episode_counts, source_frame_counts):
        ep_pq = raw_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not ep_pq.exists():
            raise FileNotFoundError(f"Missing meta episodes parquet: {ep_pq}")
        edf = pd.read_parquet(ep_pq).copy()

        edf["episode_index"] = edf["episode_index"].astype(np.int64) + global_ep_offset
        edf["dataset_from_index"] = edf["dataset_from_index"].astype(np.int64) + global_frame_offset
        edf["dataset_to_index"] = edf["dataset_to_index"].astype(np.int64) + global_frame_offset

        # video timestamp ranges become continuous over concatenated video
        if "videos/observation.images.d435/from_timestamp" in edf.columns:
            edf["videos/observation.images.d435/from_timestamp"] = (
                edf["videos/observation.images.d435/from_timestamp"].astype(float) + global_time_offset
            )
        if "videos/observation.images.d435/to_timestamp" in edf.columns:
            edf["videos/observation.images.d435/to_timestamp"] = (
                edf["videos/observation.images.d435/to_timestamp"].astype(float) + global_time_offset
            )

        ep_parts.append(edf)

        # advance offsets using source info
        src_info = json.load(open(raw_dir / "meta" / "info.json", "r", encoding="utf-8"))
        fps = float(src_info.get("fps", 30.0))
        global_ep_offset += src_ep_n
        global_frame_offset += src_frame_n
        global_time_offset += src_frame_n / fps

    merged_ep = pd.concat(ep_parts, axis=0, ignore_index=True)
    out_ep_pq = merged_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    merged_ep.to_parquet(out_ep_pq, index=False)

    # --- copy tasks.parquet from first (single task scene) ---
    first_tasks = raw_input_dirs[0] / "meta" / "tasks.parquet"
    if first_tasks.exists():
        shutil.copy2(first_tasks, merged_root / "meta" / "tasks.parquet")

    # --- merge video ---
    src_videos = [d / "videos" / "observation.images.d435" / "chunk-000" / "file-000.mp4" for d in raw_input_dirs]
    for v in src_videos:
        if not v.exists():
            raise FileNotFoundError(f"Missing raw video: {v}")
    out_video = merged_root / "videos" / "observation.images.d435" / "chunk-000" / "file-000.mp4"
    _concat_videos(src_videos, out_video)

    # --- merge depth episodes + depth.npz/index/manifest ---
    depth_cam_root = merged_root / "depth" / "d435"
    depth_ep_root = depth_cam_root / "episodes"
    depth_index_rows = []
    manifest_rows = []
    episode_meta: list[tuple[int, Path, int]] = []
    depth_cursor = 0
    out_ep_idx = 0

    for raw_dir in _iter(raw_input_dirs, "merge raw depth episodes", "src"):
        src_depth_root = raw_dir / "depth" / "d435"
        if not src_depth_root.exists():
            raise FileNotFoundError(f"Missing raw depth root: {src_depth_root}")

        # copy intrinsics from first only
        intr = src_depth_root / "intrinsics.json"
        if intr.exists() and not (depth_cam_root / "intrinsics.json").exists():
            shutil.copy2(intr, depth_cam_root / "intrinsics.json")

        src_eps = sorted((src_depth_root / "episodes").glob("episode-*.npz"))
        for ep_npz in _iter(src_eps, f"append {raw_dir.name} depth", "ep"):
            arr = np.load(ep_npz)["depth_mm"]
            out_ep = depth_ep_root / f"episode-{out_ep_idx:06d}.npz"
            np.savez_compressed(out_ep, depth_mm=arr.astype(np.uint16, copy=False))

            ep_len = int(arr.shape[0])
            start = depth_cursor
            end = depth_cursor + ep_len
            depth_index_rows.append({"episode_index": out_ep_idx, "start": start, "end": end})
            manifest_rows.append({
                "episode_index": out_ep_idx,
                "path": str(out_ep.relative_to(merged_root)),
                "frames": ep_len,
            })
            episode_meta.append((out_ep_idx, out_ep, ep_len))
            depth_cursor = end
            out_ep_idx += 1

    if episode_meta:
        # Low-memory merge: stream episode npz into a memmap, then write one canonical depth.npz.
        sample = np.load(episode_meta[0][1])["depth_mm"]
        h, w = int(sample.shape[1]), int(sample.shape[2])
        total_frames = int(depth_cursor)
        # Keep temp mmap inside output depth directory (usually on /home), avoid filling root /tmp.
        tmp_mmap_path = Path(
            tempfile.mkstemp(
                prefix="raw_depth_merge_",
                suffix=".mmap",
                dir=str(depth_cam_root),
            )[1]
        )
        try:
            merged_mm = np.memmap(tmp_mmap_path, mode="w+", dtype=np.uint16, shape=(total_frames, h, w))
            cur = 0
            for _, ep_path, ep_len in _iter(episode_meta, "build merged depth", "ep"):
                ep_arr = np.load(ep_path)["depth_mm"].astype(np.uint16, copy=False)
                merged_mm[cur : cur + ep_len] = ep_arr
                cur += ep_len
            merged_mm.flush()

            np.savez_compressed(depth_cam_root / "depth.npz", depth_mm=merged_mm)
        finally:
            try:
                tmp_mmap_path.unlink(missing_ok=True)
            except Exception:
                pass

        (depth_cam_root / "depth_index.json").write_text(
            json.dumps({"camera_key": "d435", "total_frames": int(total_frames), "episodes": depth_index_rows}, indent=2),
            encoding="utf-8",
        )
        (depth_ep_root / "manifest.json").write_text(
            json.dumps({"camera_key": "d435", "episodes": manifest_rows}, indent=2),
            encoding="utf-8",
        )

    # --- rebuild minimal info.json + stats.json ---
    first_info = json.load(open(raw_input_dirs[0] / "meta" / "info.json", "r", encoding="utf-8"))
    merged_info = dict(first_info)
    merged_info["total_episodes"] = int(ep_cursor)
    merged_info["total_frames"] = int(len(merged_data))
    # keep path template and features from first source
    (merged_root / "meta" / "info.json").write_text(json.dumps(merged_info, indent=2), encoding="utf-8")

    first_stats = raw_input_dirs[0] / "meta" / "stats.json"
    if first_stats.exists():
        shutil.copy2(first_stats, merged_root / "meta" / "stats.json")

    print(f"[INFO] raw merged: {merged_root}")
    print(f"[INFO] raw total episodes: {ep_cursor}")
    print(f"[INFO] raw total frames: {len(merged_data)}")


def merge_hamer_packages(
    input_dirs: list[Path],
    output_dir: Path,
    merged_name: str,
    keep_nonmerged: bool = True,
    merge_raw: bool = True,
    raw_input_dirs: list[Path] | None = None,
    raw_output_dir: Path | None = None,
    raw_root: Path | None = None,
) -> None:
    output_root = output_dir / merged_name
    merged_seq_name = f"{merged_name}_d435_chunk-000_file-000"
    output_seq_dir = output_root / merged_seq_name
    output_results_dir = output_seq_dir / "results"
    output_results_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths: list[Path] = []
    render_paths: list[Path] = []
    source_names: list[str] = []

    for ds_dir in _iter(input_dirs, "scan datasets", "ds"):
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {ds_dir}")
        seq_dirs = [p for p in ds_dir.iterdir() if p.is_dir()]
        if len(seq_dirs) != 1:
            raise ValueError(f"Expect exactly one sequence dir under {ds_dir}, got {len(seq_dirs)}")
        seq_dir = seq_dirs[0]

        pkl_candidates = sorted(seq_dir.glob("*.pkl"))
        if len(pkl_candidates) != 1:
            raise ValueError(f"Expect exactly one pkl in {seq_dir}, got {len(pkl_candidates)}")
        pkl_path = pkl_candidates[0]

        render_path = seq_dir / "results" / "render_all_500.0.mp4"
        if not render_path.exists():
            raise FileNotFoundError(f"Missing render video: {render_path}")

        source_names.append(seq_dir.name)
        pkl_paths.append(pkl_path)
        render_paths.append(render_path)

    payloads = [_load_pkl(p) for p in _iter(pkl_paths, "load pkl", "file")]
    merged_pkl = _merge_pkl_payloads(payloads, source_names, merged_seq_name)

    out_pkl = output_seq_dir / f"{merged_seq_name}.pkl"
    _save_pkl(out_pkl, merged_pkl)

    out_render = output_results_dir / "render_all_500.0.mp4"
    _concat_videos(render_paths, out_render)

    copied_files = []
    if keep_nonmerged:
        for ds_dir in input_dirs:
            seq_dir = next(p for p in ds_dir.iterdir() if p.is_dir())
            src_results = seq_dir / "results"
            if not src_results.exists():
                continue
            dst_ns = output_results_dir / "sources" / seq_dir.name
            dst_ns.mkdir(parents=True, exist_ok=True)
            for f in src_results.rglob("*"):
                if f.is_file() and f.name != "render_all_500.0.mp4":
                    rel = f.relative_to(src_results)
                    dst = dst_ns / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    copied_files.append(str(dst))

    print(f"[INFO] merged datasets: {len(input_dirs)}")
    print(f"[INFO] output sequence: {merged_seq_name}")
    print(f"[INFO] wrote pkl: {out_pkl}")
    print(f"[INFO] wrote video: {out_render}")

    if merge_raw:
        # Priority:
        # 1) explicit raw_input_dirs/raw_output_dir
        # 2) raw_root + infer by dataset name
        # 3) default sibling raw dir
        if raw_input_dirs is not None and len(raw_input_dirs) > 0:
            raw_inputs = raw_input_dirs
        else:
            if raw_root is None:
                # default: sibling raw dir under same data root as output_dir
                raw_root = output_dir.parent / "raw"
            raw_inputs = [raw_root / d.name for d in input_dirs]

        if raw_output_dir is None:
            if raw_root is not None:
                raw_output_dir = raw_root
            else:
                raw_output_dir = output_dir.parent / "raw"

        for r in raw_inputs:
            if not r.exists():
                raise FileNotFoundError(f"Expected raw dataset not found for {r.name}: {r}")
        _merge_raw_packages(raw_inputs, raw_output_dir, merged_name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge full HaMeR output packages (pkl + render video + optional raw merge).")
    ap.add_argument(
        "--hamer-input-dirs",
        "--input-dirs",
        dest="hamer_input_dirs",
        nargs="+",
        required=True,
        help="Input dataset dirs under hamer_outputs (at least 2)",
    )
    ap.add_argument("--output-dir", required=True, help="Output parent dir, e.g. data/hamer_outputs")
    ap.add_argument("--merged-name", required=True, help="Merged dataset name, e.g. 5.24_merged_bana")
    ap.add_argument("--no-copy-extra-results", action="store_true", help="Do not copy non-render result files")
    ap.add_argument("--no-merge-raw", action="store_true", help="Skip raw dataset merge")
    ap.add_argument(
        "--raw-input-dirs",
        nargs="+",
        default=None,
        help="Optional explicit raw input dataset dirs. If omitted, infer from --raw-root and input dir names.",
    )
    ap.add_argument(
        "--raw-output-dir",
        default=None,
        help="Optional raw output parent dir. Default: --raw-root (or <output-dir>/../raw).",
    )
    ap.add_argument("--raw-root", default=None, help="Raw dataset root, default: <output-dir>/../raw")
    args = ap.parse_args()

    input_dirs = [Path(x).expanduser().resolve() for x in args.hamer_input_dirs]
    if len(input_dirs) < 2:
        raise ValueError("Need at least two input dirs.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_input_dirs = [Path(x).expanduser().resolve() for x in args.raw_input_dirs] if args.raw_input_dirs else None
    raw_output_dir = Path(args.raw_output_dir).expanduser().resolve() if args.raw_output_dir else None
    raw_root = Path(args.raw_root).expanduser().resolve() if args.raw_root else None

    merge_hamer_packages(
        input_dirs=input_dirs,
        output_dir=output_dir,
        merged_name=args.merged_name,
        keep_nonmerged=not args.no_copy_extra_results,
        merge_raw=not args.no_merge_raw,
        raw_input_dirs=raw_input_dirs,
        raw_output_dir=raw_output_dir,
        raw_root=raw_root,
    )


if __name__ == "__main__":
    main()
