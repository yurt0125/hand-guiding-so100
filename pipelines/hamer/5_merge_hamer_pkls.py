#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _iter_with_progress(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="item")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def _merge_list_payload(payloads: list[list[Any]], src_names: list[str]) -> list[Any]:
    merged: list[Any] = []
    offset = 0
    for i, frames in enumerate(_iter_with_progress(payloads, "merge payloads")):
        for rec in _iter_with_progress(frames, f"append {src_names[i]}"):
            # best-effort: keep source trace + global frame index
            if isinstance(rec, dict):
                rec = dict(rec)
                rec.setdefault("source", src_names[i])
                rec["merged_index"] = offset
            merged.append(rec)
            offset += 1
    return merged


def _merge_dict_results(payloads: list[dict[str, Any]], src_names: list[str]) -> dict[str, Any]:
    merged = dict(payloads[0])
    merged_results: list[Any] = []
    source_ranges = []
    cursor = 0

    for i, payload in enumerate(_iter_with_progress(payloads, "merge payloads")):
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Payload {src_names[i]} dict has no list 'results' key")

        start = cursor
        for rec in _iter_with_progress(results, f"append {src_names[i]}"):
            if isinstance(rec, dict):
                rec = dict(rec)
                rec.setdefault("source", src_names[i])
                rec["merged_index"] = cursor
            merged_results.append(rec)
            cursor += 1
        end = cursor
        source_ranges.append({"source": src_names[i], "start": start, "end": end})

    merged["results"] = merged_results
    merged["source_ranges"] = source_ranges
    merged["total_frames"] = len(merged_results)
    return merged


def merge_hamer_pkls(input_pkls: list[Path], output_pkl: Path) -> None:
    payloads = []
    src_names = [p.stem for p in input_pkls]

    for p in _iter_with_progress(input_pkls, "load pkls"):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        payloads.append(_load_pickle(p))

    first = payloads[0]

    if isinstance(first, list):
        for i, x in enumerate(payloads):
            if not isinstance(x, list):
                raise TypeError(f"Mixed payload types: #{i} is {type(x)} while first is list")
        merged = _merge_list_payload(payloads, src_names)
    elif isinstance(first, dict):
        for i, x in enumerate(payloads):
            if not isinstance(x, dict):
                raise TypeError(f"Mixed payload types: #{i} is {type(x)} while first is dict")
        merged = _merge_dict_results(payloads, src_names)
    else:
        raise TypeError(f"Unsupported payload type: {type(first)}")

    _save_pickle(output_pkl, merged)

    total = len(merged) if isinstance(merged, list) else int(merged.get("total_frames", -1))
    print(f"[INFO] merged inputs: {len(input_pkls)}")
    print(f"[INFO] total frames: {total}")
    print(f"[INFO] wrote: {output_pkl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple Dyn-HaMR/HaMeR pkl files into one pkl with progress bars.")
    parser.add_argument(
        "--input-pkls",
        nargs="+",
        required=True,
        help="Input pkl paths (at least 2).",
    )
    parser.add_argument("--output-pkl", required=True, help="Output merged pkl path.")
    args = parser.parse_args()

    input_pkls = [Path(x).expanduser().resolve() for x in args.input_pkls]
    if len(input_pkls) < 2:
        raise ValueError("Please provide at least two input pkls.")

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    merge_hamer_pkls(input_pkls=input_pkls, output_pkl=output_pkl)


if __name__ == "__main__":
    main()
