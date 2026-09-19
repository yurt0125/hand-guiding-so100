#!/usr/bin/env python3
"""Replace the sole task string in an already-created LeRobot taskfix dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: set_taskfix_text.py DATASET_DIR TASK_TEXT")
    dataset_dir = Path(sys.argv[1])
    task_text = sys.argv[2]
    tasks_path = dataset_dir / "meta/tasks.parquet"
    episodes_path = dataset_dir / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(episodes_path)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task_text], name="task")).to_parquet(tasks_path)
    episodes["tasks"] = [[task_text] for _ in range(len(episodes))]
    episodes.to_parquet(episodes_path, index=False)
    print(f"updated {dataset_dir}: {len(episodes)} episodes -> {task_text}")


if __name__ == "__main__":
    main()
