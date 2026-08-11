#!/usr/bin/env python3
"""Unified launcher for SO100 runtime modes."""

from __future__ import annotations

import argparse
import os
import sys

# These must be set before importing MuJoCo/CasADi/NumPy-heavy runtime modules.
# If they are set after those imports, replay IK can jump from ~1 ms to hundreds
# of milliseconds on some machines due to thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO100 runtime mode launcher")
    parser.add_argument(
        "--mode",
        choices=["keyboard", "mujoco", "keyboard_mirror", "vision", "replay"],
        required=True,
        help="Runtime mode to start",
    )
    parser.add_argument("--csv-path", default=None, help="Replay CSV path (used in --mode replay)")
    parser.add_argument("--port", default=None, help="Robot USB port (used in --mode replay)")
    parser.add_argument("--fps", type=float, default=None, help="Replay FPS (used in --mode replay)")
    parser.add_argument(
        "--export-action-csv",
        default=None,
        help="Optional path to export replay actual sent actions (used in --mode replay)",
    )
    parser.add_argument(
        "--target-offset-x",
        type=float,
        default=None,
        help="Replay target X offset in meters (used in --mode replay)",
    )
    parser.add_argument(
        "--target-offset-y",
        type=float,
        default=None,
        help="Replay target Y offset in meters (used in --mode replay)",
    )
    parser.add_argument(
        "--target-offset-z",
        type=float,
        default=None,
        help="Replay target Z offset in meters (used in --mode replay)",
    )
    parser.add_argument(
        "--episode-reset-time-s",
        type=float,
        default=None,
        help="Replay pause duration in seconds between episodes (used in --mode replay)",
    )
    parser.add_argument(
        "--record-video-path",
        default=None,
        help="Optional MP4 output path to record camera video during replay (used in --mode replay)",
    )
    parser.add_argument(
        "--record-rs-serial",
        default=None,
        help="Optional RealSense serial for replay recording (used in --mode replay)",
    )
    parser.add_argument(
        "--record-video-width",
        type=int,
        default=None,
        help="Replay recording width (used in --mode replay)",
    )
    parser.add_argument(
        "--record-video-height",
        type=int,
        default=None,
        help="Replay recording height (used in --mode replay)",
    )
    parser.add_argument(
        "--record-video-fps",
        type=float,
        default=None,
        help="Replay recording FPS (used in --mode replay)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume replay recording from existing exports (used in --mode replay)",
    )
    parser.add_argument(
        "--profile-control",
        action="store_true",
        help="Print replay timing diagnostics once per second (used in --mode replay)",
    )
    parser.add_argument(
        "--no-auto-resync-on-jump",
        action="store_true",
        help="Keep old replay behavior and skip large joint jumps instead of rate-limited resync.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.mode == "keyboard":
        from app.keyboard_mirror_app import main_real_only as keyboard_real_only_main

        keyboard_real_only_main(port=arguments.port)
        return
    if arguments.mode == "mujoco":
        from app.keyboard_mirror_app import main_sim_only as keyboard_mujoco_main

        keyboard_mujoco_main()
        return
    if arguments.mode == "keyboard_mirror":
        from app.keyboard_mirror_app import main as keyboard_mirror_main

        keyboard_mirror_main(port=arguments.port)
        return
    if arguments.mode == "vision":
        from app.vision_app_refactored import main as vision_main

        vision_main()
        return
    # Explicitly forward replay args so offsets/paths always take effect.
    replay_argv = [sys.argv[0]]
    if arguments.csv_path is not None:
        replay_argv += ["--csv-path", str(arguments.csv_path)]
    if arguments.port is not None:
        replay_argv += ["--port", str(arguments.port)]
    if arguments.fps is not None:
        replay_argv += ["--fps", str(arguments.fps)]
    if arguments.export_action_csv is not None:
        replay_argv += ["--export-action-csv", str(arguments.export_action_csv)]
    if arguments.target_offset_x is not None:
        replay_argv += ["--target-offset-x", str(arguments.target_offset_x)]
    if arguments.target_offset_y is not None:
        replay_argv += ["--target-offset-y", str(arguments.target_offset_y)]
    if arguments.target_offset_z is not None:
        replay_argv += ["--target-offset-z", str(arguments.target_offset_z)]
    if arguments.episode_reset_time_s is not None:
        replay_argv += ["--episode-reset-time-s", str(arguments.episode_reset_time_s)]
    if arguments.record_video_path is not None:
        replay_argv += ["--record-video-path", str(arguments.record_video_path)]
    if arguments.record_rs_serial is not None:
        replay_argv += ["--record-rs-serial", str(arguments.record_rs_serial)]
    if arguments.record_video_width is not None:
        replay_argv += ["--record-video-width", str(arguments.record_video_width)]
    if arguments.record_video_height is not None:
        replay_argv += ["--record-video-height", str(arguments.record_video_height)]
    if arguments.record_video_fps is not None:
        replay_argv += ["--record-video-fps", str(arguments.record_video_fps)]
    if arguments.resume:
        replay_argv += ["--resume"]
    if arguments.profile_control:
        replay_argv += ["--profile-control"]
    if arguments.no_auto_resync_on_jump:
        replay_argv += ["--no-auto-resync-on-jump"]
    sys.argv = replay_argv
    from app.replay_app_refactored import main as replay_main

    replay_main()


if __name__ == "__main__":
    main()
