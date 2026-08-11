#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_step(command_arguments: list[str]) -> None:
    command_text = " ".join(shlex.quote(argument) for argument in command_arguments)
    print(f"[RUN] {command_text}")
    subprocess.run(command_arguments, check=True)


def default_scored_csv(window_json_path: Path) -> Path:
    return window_json_path.with_suffix(".scored.csv")


def resolve_pipeline_paths(arguments: argparse.Namespace, project_root: Path) -> tuple[Path, Path, Path, Path]:
    if arguments.episode_name:
        episode_name = arguments.episode_name.strip()
        if not episode_name:
            raise ValueError("--episode-name cannot be empty")
        input_csv_default = project_root / "data" / "replay_csv" / f"{episode_name}_demo_smoothed.csv"
        grasp_data_default = project_root / "third-party" / "graspnet-baseline" / "doc" / "my_grasp_data"
        checkpoint_default = project_root / "third-party" / "graspnet-baseline" / "logs" / "log_rs" / "checkpoint-rs.tar"
        output_prefix_default = project_root / "artifacts" / "grasp_prior" / episode_name
    else:
        input_csv_default = None
        grasp_data_default = None
        checkpoint_default = None
        output_prefix_default = None

    input_csv_raw = arguments.input_csv if arguments.input_csv else input_csv_default
    grasp_data_raw = arguments.grasp_data_dir if arguments.grasp_data_dir else grasp_data_default
    checkpoint_raw = arguments.checkpoint_path if arguments.checkpoint_path else checkpoint_default
    output_prefix_raw = arguments.output_prefix if arguments.output_prefix else output_prefix_default

    if input_csv_raw is None:
        raise ValueError("Missing --input-csv. You can also provide --episode-name to use default paths.")
    if grasp_data_raw is None:
        raise ValueError("Missing --grasp-data-dir. You can also provide --episode-name to use default paths.")
    if checkpoint_raw is None:
        raise ValueError("Missing --checkpoint-path. You can also provide --episode-name to use default paths.")
    if output_prefix_raw is None:
        raise ValueError("Missing --output-prefix. You can also provide --episode-name to use default paths.")

    input_csv_path = Path(input_csv_raw).expanduser().resolve()
    grasp_data_dir_path = Path(grasp_data_raw).expanduser().resolve()
    checkpoint_path = Path(checkpoint_raw).expanduser().resolve()
    output_prefix_path = Path(output_prefix_raw).expanduser().resolve()
    return input_csv_path, grasp_data_dir_path, checkpoint_path, output_prefix_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grasp prior -> window detect -> anchor fuse -> VLA export -> validation in one command."
    )
    parser.add_argument("--episode-name", default=None, help="Shortcut name, e.g. 4.16")
    parser.add_argument("--input-csv", default=None, help="Smoothed trajectory CSV")
    parser.add_argument(
        "--grasp-data-dir",
        default=None,
        help="GraspNet demo folder with color.png/depth.png/workspace_mask.png/meta.mat",
    )
    parser.add_argument("--checkpoint-path", default=None, help="GraspNet checkpoint path")
    parser.add_argument("--output-prefix", default=None, help="Output prefix, e.g. artifacts/grasp_prior/4.16")
    parser.add_argument(
        "--transform-json",
        default=None,
        help="Optional camera-to-base transform JSON path (default follows project config)",
    )
    parser.add_argument("--chunk-size", type=int, default=10, help="VLA export chunk size")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback fps for VLA export")
    parser.add_argument("--alpha-max", type=float, default=0.45)
    parser.add_argument("--alpha-profile", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--min-score-to-fuse", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug in grasp inference")
    parser.add_argument("--overwrite-orientation", action="store_true", help="Blend roll/pitch/yaw in fusion")
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    python_executable = sys.executable

    input_csv_path, grasp_data_dir_path, checkpoint_path, output_prefix_path = resolve_pipeline_paths(
        arguments=arguments,
        project_root=project_root,
    )
    output_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    grasp_prior_json_path = Path(f"{output_prefix_path}_top_grasp.json")
    grasp_window_json_path = Path(f"{output_prefix_path}_grasp_window.json")
    grasp_scored_csv_path = default_scored_csv(grasp_window_json_path)
    fused_csv_path = Path(f"{output_prefix_path}_fused.csv")
    vla_csv_path = Path(f"{output_prefix_path}_vla.csv")
    vla_npz_path = Path(f"{output_prefix_path}_vla.npz")

    infer_command = [
        python_executable,
        str(project_root / "pipelines/grasp/5_infer_grasp_prior.py"),
        "--data-dir",
        str(grasp_data_dir_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-json",
        str(grasp_prior_json_path),
    ]
    if arguments.transform_json:
        infer_command.extend(["--transform-json", str(Path(arguments.transform_json).expanduser().resolve())])
    if arguments.debug:
        infer_command.append("--debug")
    run_step(infer_command)

    run_step(
        [
            python_executable,
            str(project_root / "pipelines/grasp/6_detect_grasp_window.py"),
            "--input-csv",
            str(input_csv_path),
            "--output-json",
            str(grasp_window_json_path),
            "--output-scored-csv",
            str(grasp_scored_csv_path),
        ]
    )

    fuse_command = [
        python_executable,
        str(project_root / "pipelines/grasp/7_fuse_grasp_anchor.py"),
        "--input-csv",
        str(input_csv_path),
        "--grasp-prior-json",
        str(grasp_prior_json_path),
        "--grasp-window-json",
        str(grasp_window_json_path),
        "--output-csv",
        str(fused_csv_path),
        "--alpha-max",
        str(arguments.alpha_max),
        "--alpha-profile",
        arguments.alpha_profile,
        "--min-score-to-fuse",
        str(arguments.min_score_to_fuse),
    ]
    if arguments.overwrite_orientation:
        fuse_command.append("--overwrite-orientation")
    run_step(fuse_command)

    for vla_output_path, output_format in [(vla_csv_path, "csv"), (vla_npz_path, "npz")]:
        run_step(
            [
                python_executable,
                str(project_root / "pipelines/export/export_vla_dataset.py"),
                "--input-csv",
                str(fused_csv_path),
                "--output-csv",
                str(vla_output_path),
                "--output-format",
                output_format,
                "--chunk-size",
                str(arguments.chunk_size),
                "--fps",
                str(arguments.fps),
            ]
        )
        run_step(
            [
                python_executable,
                str(project_root / "pipelines/export/validate_vla_dataset.py"),
                "--input-path",
                str(vla_output_path),
            ]
        )

    print("[DONE] Grasp fusion pipeline completed.")
    print(f"[OUT] grasp_prior_json={grasp_prior_json_path}")
    print(f"[OUT] grasp_window_json={grasp_window_json_path}")
    print(f"[OUT] grasp_scored_csv={grasp_scored_csv_path}")
    print(f"[OUT] fused_csv={fused_csv_path}")
    print(f"[OUT] vla_csv={vla_csv_path}")
    print(f"[OUT] vla_npz={vla_npz_path}")


if __name__ == "__main__":
    main()
