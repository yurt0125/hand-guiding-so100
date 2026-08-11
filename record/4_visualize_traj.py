#!/usr/bin/env python3
"""Compatibility wrapper for the relocated trajectory visualization pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "pipelines" / "visualize" / "4_visualize_traj.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
