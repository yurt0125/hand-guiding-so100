#!/usr/bin/env python3
"""Compatibility wrapper for the relocated recorder pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "pipelines" / "record" / "1_d435_episode_record.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
