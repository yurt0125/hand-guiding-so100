#!/usr/bin/env python3
"""Clone selected LeRobot datasets and replace their single task text.

The source datasets are never modified.  Frame-level task_index remains 0;
only the task lookup table and per-episode task lists need changing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path("/home/robot/.cache/huggingface/lerobot/RITAHuang")
TASKS = {
    "blocks": "Put the green block in the basket",
    "push": "Push the green block to the target area",
    "pull": "Pull the storage box",
    "stacks": "Stack the blocks",
    "hanging": "Hang the tape",
}


def main() -> None:
    for name, task_text in TASKS.items():
        source = ROOT / f"smolvla_{name}_0_49_merged"
        target = ROOT / f"smolvla_{name}_0_49_merged_taskfix"
        tasks_path = target / "meta/tasks.parquet"
        episodes_path = target / "meta/episodes/chunk-000/file-000.parquet"
        if target.exists():
            existing_tasks = pd.read_parquet(tasks_path).index.tolist()
            episodes = pd.read_parquet(episodes_path)
            if existing_tasks == [task_text] and all(item == [task_text] for item in episodes["tasks"]):
                print(f"verified existing {target} ({len(episodes)} episodes): {task_text}")
                continue
            raise RuntimeError(f"Existing dataset does not match expected task text: {target}")

        shutil.copytree(source, target, copy_function=shutil.copy2)
        task_table = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_text], name="task"))
        task_table.to_parquet(tasks_path)

        episodes = pd.read_parquet(episodes_path)
        episodes["tasks"] = [[task_text] for _ in range(len(episodes))]
        episodes.to_parquet(episodes_path, index=False)
        print(f"created {target} ({len(episodes)} episodes): {task_text}")


if __name__ == "__main__":
    main()
