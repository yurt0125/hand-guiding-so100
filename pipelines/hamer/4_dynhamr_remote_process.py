#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/rita/hand-guiding-so100-master")


def run_command(command: list[str]) -> None:
    print(f"[CMD] {' '.join(command)}")
    subprocess.run(command, check=True)


def run_command_with_return_code(command: list[str]) -> int:
    print(f"[CMD] {' '.join(command)}")
    result = subprocess.run(command, check=False)
    return result.returncode


def build_ssh_command(
    sshpass_prefix: list[str],
    remote_login: str,
    remote_script: str,
    use_alias: bool,
    server_port: int,
) -> list[str]:
    command = sshpass_prefix + ["ssh", remote_login, remote_script]
    if not use_alias:
        command[2:2] = ["-p", str(server_port)]
    return command


def build_ssh_prefix(password: str | None) -> list[str]:
    if not password:
        return []
    sshpass_path = shutil.which("sshpass")
    if sshpass_path is None:
        print(
            "[WARN] Password env provided but sshpass is not installed. "
            "Falling back to interactive password input."
        )
        return []
    return [sshpass_path, "-p", password]


def ensure_local_output_dir(local_output_dir: Path) -> None:
    local_output_dir.mkdir(parents=True, exist_ok=True)


def run_single_job(
    *,
    video_path: Path,
    sequence_name: str,
    server_host: str,
    server_port: int,
    server_user: str,
    server_alias: str | None,
    remote_project_root: str,
    remote_test_root: str,
    remote_video_dir: str,
    local_output_dir: Path,
    conda_env: str,
    use_vipe: bool,
    gpu_id: int | None,
    dynhamr_fps: int,
    sshpass_prefix: list[str],
    skip_existing: bool,
    download_full_results: bool,
) -> bool:
    if not video_path.exists():
        print(f"[SKIP] video not found: {video_path}")
        return False
    if video_path.suffix.lower() != ".mp4":
        print(f"[SKIP] not mp4: {video_path}")
        return False

    use_alias = bool(server_alias)
    remote_login = server_alias if use_alias else f"{server_user}@{server_host}"
    remote_video_path = f"{remote_video_dir}/{sequence_name}.mp4"
    remote_result_dir = f"{remote_test_root}/dynhamr/hamer_out/{sequence_name}"
    remote_images_dir = f"{remote_test_root}/images/{sequence_name}"
    remote_cameras_dir = f"{remote_test_root}/dynhamr/cameras/{sequence_name}"
    remote_tracks_dir = f"{remote_test_root}/dynhamr/track_preds/{sequence_name}"
    remote_shots_json = f"{remote_test_root}/dynhamr/shot_idcs/{sequence_name}.json"
    local_result_dir = local_output_dir / sequence_name
    local_pkl_path = local_result_dir / f"{sequence_name}.pkl"

    # Skip only when the required artifact really exists and is non-empty.
    if skip_existing and local_pkl_path.exists() and local_pkl_path.stat().st_size > 0:
        print(f"[SKIP] existing local output: {local_pkl_path}")
        return True

    print(f"\n[JOB] {sequence_name}")
    print(f"[INFO] video: {video_path}")

    print("[INFO] Step 1/5: upload video")
    upload_command = (
        sshpass_prefix
        + [
            "scp",
            str(video_path),
            f"{remote_login}:{remote_video_path}",
        ]
    )
    if not use_alias:
        upload_command[2:2] = ["-P", str(server_port)]
    run_command(upload_command)

    print("[INFO] Step 2/5: check existing remote result")
    remote_check_script = f"test -f {remote_result_dir}/{sequence_name}.pkl"
    remote_check_command = build_ssh_command(
        sshpass_prefix=sshpass_prefix,
        remote_login=remote_login,
        remote_script=remote_check_script,
        use_alias=use_alias,
        server_port=server_port,
    )
    remote_result_exists = run_command_with_return_code(remote_check_command) == 0
    if remote_result_exists:
        print("[INFO] remote result exists, skip remote inference and download directly")

    print("[INFO] Step 3/5: run Dyn-HaMR on remote")
    data_preset = "video_vipe" if use_vipe else "video_driod"
    gpu_export = f"export CUDA_VISIBLE_DEVICES={gpu_id}; " if gpu_id is not None else ""
    gpu_override = f" gpu={gpu_id}" if gpu_id is not None else ""

    remote_run_script = (
        "set -e; "
        "if command -v conda >/dev/null 2>&1; then "
        "  eval \"$(conda shell.bash hook)\"; "
        "elif [ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then "
        "  source \"$HOME/miniconda3/etc/profile.d/conda.sh\"; "
        "elif [ -f \"$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then "
        "  source \"$HOME/anaconda3/etc/profile.d/conda.sh\"; "
        "else "
        "  echo '[ERROR] conda init script not found on remote host'; "
        "  exit 127; "
        "fi; "
        f"conda activate {conda_env}; "
        f"cd {remote_project_root}; "
        "if [ -f third-party/hamer/run2.py ]; then "
        "  sed -i 's/python -u run.py/python -u run2.py/g' dyn-hamr/preproc/launch_hamer.py; "
        "  echo '[INFO] force right-hand-only: launch_hamer.py -> run2.py'; "
        "else "
        "  echo '[WARN] third-party/hamer/run2.py not found, fallback to run.py'; "
        "fi; "
        f"cd {remote_project_root}/dyn-hamr; "
        f"{gpu_export}"
        "export PYTHONPATH=.; "
        "python run_opt.py "
        f"data={data_preset} run_opt=True data.seq={sequence_name} "
        f"data.root={remote_test_root} is_static=False"
        f" data.frame_opts.fps={dynhamr_fps}"
        f"{gpu_override}"
    )
    if not remote_result_exists:
        remote_command = build_ssh_command(
            sshpass_prefix=sshpass_prefix,
            remote_login=remote_login,
            remote_script=remote_run_script,
            use_alias=use_alias,
            server_port=server_port,
        )
        remote_return_code = run_command_with_return_code(remote_command)
        if remote_return_code != 0:
            print(f"[WARN] remote run returned non-zero exit code: {remote_return_code}")
            print("[WARN] continue to pull artifacts if they were partially generated")

    print("[INFO] Step 4/5: sync key artifacts back to local")
    if local_result_dir.exists():
        shutil.rmtree(local_result_dir)
    local_result_dir.mkdir(parents=True, exist_ok=True)

    def _download(remote_file: str, local_file: Path, recursive: bool = False) -> bool:
        local_file.parent.mkdir(parents=True, exist_ok=True)
        command = sshpass_prefix + ["scp"]
        if recursive:
            command.append("-r")
        command += [f"{remote_login}:{remote_file}", str(local_file)]
        if not use_alias:
            command[2:2] = ["-P", str(server_port)]
        try:
            run_command(command)
            return True
        except subprocess.CalledProcessError:
            return False

    remote_pkl_path = f"{remote_result_dir}/{sequence_name}.pkl"
    remote_render_mp4_path = f"{remote_result_dir}/results/render_all_500.0.mp4"
    local_render_mp4_path = local_result_dir / "results" / "render_all_500.0.mp4"

    pkl_ok = _download(remote_pkl_path, local_pkl_path)
    if not pkl_ok:
        print("[WARN] failed to download required pkl")
        download_ok = False
    else:
        download_ok = True

    render_ok = _download(remote_render_mp4_path, local_render_mp4_path)
    if not render_ok:
        print("[WARN] render_all_500.0.mp4 not found or download failed (optional)")

    if download_ok and download_full_results:
        full_results_ok = _download(f"{remote_result_dir}/results", local_result_dir / "results", recursive=True)
        if not full_results_ok:
            print("[WARN] failed to download full results directory")

    if download_ok:
        expected_local_pkl = local_result_dir / f"{sequence_name}.pkl"
        if not expected_local_pkl.exists():
            print(f"[WARN] expected pkl not found after sync: {expected_local_pkl}")
            print("[WARN] treat this job as failed; keep remote artifacts for debugging")
            return False

        print("[INFO] Step 5/5: cleanup remote artifacts")
        remote_cleanup_script = (
            f"rm -rf {remote_result_dir} "
            f"{remote_images_dir} "
            f"{remote_cameras_dir} "
            f"{remote_tracks_dir} && "
            f"rm -f {remote_video_path} {remote_shots_json}"
        )
        cleanup_command = build_ssh_command(
            sshpass_prefix=sshpass_prefix,
            remote_login=remote_login,
            remote_script=remote_cleanup_script,
            use_alias=use_alias,
            server_port=server_port,
        )
        cleanup_rc = run_command_with_return_code(cleanup_command)
        if cleanup_rc != 0:
            print(f"[WARN] remote cleanup failed with exit code: {cleanup_rc}")

    print("[INFO] Step 6/6: done")
    print(f"[INFO] Sequence: {sequence_name}")
    if download_ok:
        print(f"[INFO] Local output: {local_result_dir}")
    else:
        print("[INFO] Local output: <not downloaded>")

    return download_ok


def discover_episode_jobs(episodes_root: Path, include_d435: bool, include_top: bool) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for episode_dir in sorted(episodes_root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        episode_name = episode_dir.name

        if include_d435:
            d435_video_path = episode_dir / "d435" / "rgb.mp4"
            if d435_video_path.exists():
                jobs.append((f"{episode_name}_d435", d435_video_path))

        if include_top:
            top_video_path = episode_dir / "top" / "rgb.mp4"
            if top_video_path.exists():
                jobs.append((f"{episode_name}_top", top_video_path))
    return jobs


def discover_lerobot_jobs(dataset_root: Path, include_d435: bool, include_top: bool) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        return jobs

    camera_to_folder = {
        "d435": "observation.images.d435",
        "top": "observation.images.fixed",
    }

    enabled_cameras: list[str] = []
    if include_d435:
        enabled_cameras.append("d435")
    if include_top:
        enabled_cameras.append("top")

    for camera_name in enabled_cameras:
        folder_name = camera_to_folder[camera_name]
        camera_root = videos_root / folder_name
        if not camera_root.exists():
            continue

        for video_path in sorted(camera_root.glob("chunk-*/file-*.mp4")):
            try:
                chunk_name = video_path.parent.name
                file_name = video_path.stem
                sequence_name = f"{dataset_root.name}_{camera_name}_{chunk_name}_{file_name}"
                jobs.append((sequence_name, video_path))
            except Exception:
                continue

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload video(s) to remote Dyn-HaMR server, run inference, and sync results back.",
    )
    parser.add_argument("--video-path", default=None, help="Single local video path, e.g. /.../4.16.mp4")
    parser.add_argument("--sequence-name", default=None, help="Dyn-HaMR sequence name. Default: video file stem")

    parser.add_argument("--episodes-root", default=None, help="Batch mode: directory containing episode_xxxxxx")
    parser.add_argument("--only-d435", action="store_true", help="Batch mode: process only d435/rgb.mp4")
    parser.add_argument("--only-top", action="store_true", help="Batch mode: process only top/rgb.mp4")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip if local output already exists (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Do not skip existing local outputs; force re-run/download.",
    )

    parser.add_argument("--server-host", default="10.24.11.16")
    parser.add_argument("--server-port", type=int, default=2223)
    parser.add_argument("--server-user", default="yyl")
    parser.add_argument(
        "--server-alias",
        default="yyl",
        help="SSH host alias from ~/.ssh/config. Set empty string to disable alias mode.",
    )
    parser.add_argument("--remote-project-root", default="/data2/hrd/Dyn-HaMR")
    parser.add_argument("--remote-test-root", default="/data2/hrd/Dyn-HaMR/test")
    parser.add_argument("--remote-video-dir", default="/data2/hrd/Dyn-HaMR/test/videos")
    parser.add_argument(
        "--local-output-dir",
        default=str(PROJECT_ROOT / "data/hamer_outputs"),
        help="Where to store synced Dyn-HaMR outputs locally.",
    )
    parser.add_argument("--conda-env", default="dynhamr", help="Conda env name on remote server.")
    parser.add_argument(
        "--use-vipe",
        action="store_true",
        default=True,
        help="Enable VIPE preprocessing. Default is enabled (matches legacy flow).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Remote GPU id, e.g. 1. If omitted, remote default CUDA device is used.",
    )
    parser.add_argument(
        "--dynhamr-fps",
        type=int,
        default=30,
        help="Frame sampling FPS used by Dyn-HaMR preprocessing. Set to your recording FPS.",
    )
    parser.add_argument(
        "--download-full-results",
        action="store_true",
        help="Also download full remote results directory (bbox_vis_500.0, render_all_500.0 frames, etc).",
    )
    parser.add_argument(
        "--password-env",
        default="DYNHAMR_SERVER_PASSWORD",
        help="Environment variable name for server password.",
    )
    args = parser.parse_args()

    if args.video_path and args.episodes_root:
        raise ValueError("Use either --video-path or --episodes-root, not both.")
    if not args.video_path and not args.episodes_root:
        raise ValueError("One of --video-path or --episodes-root is required.")

    if args.only_d435 and args.only_top:
        raise ValueError("--only-d435 and --only-top cannot be used together.")

    local_output_dir = Path(args.local_output_dir).expanduser().resolve()
    ensure_local_output_dir(local_output_dir)

    password = os.getenv(args.password_env)
    sshpass_prefix = build_ssh_prefix(password)

    if args.video_path:
        video_path = Path(args.video_path).expanduser().resolve()
        sequence_name = args.sequence_name or video_path.stem
        run_single_job(
            video_path=video_path,
            sequence_name=sequence_name,
            server_host=args.server_host,
            server_port=args.server_port,
            server_user=args.server_user,
            server_alias=args.server_alias or None,
            remote_project_root=args.remote_project_root,
            remote_test_root=args.remote_test_root,
            remote_video_dir=args.remote_video_dir,
            local_output_dir=local_output_dir,
            conda_env=args.conda_env,
            use_vipe=args.use_vipe,
            gpu_id=args.gpu_id,
            dynhamr_fps=args.dynhamr_fps,
            sshpass_prefix=sshpass_prefix,
            skip_existing=args.skip_existing,
            download_full_results=args.download_full_results,
        )
        return

    episodes_root = Path(args.episodes_root).expanduser().resolve()
    if not episodes_root.exists():
        raise FileNotFoundError(f"episodes root not found: {episodes_root}")

    include_d435 = not args.only_top
    include_top = not args.only_d435

    jobs = discover_episode_jobs(episodes_root, include_d435=include_d435, include_top=include_top)
    if not jobs:
        jobs = discover_lerobot_jobs(episodes_root, include_d435=include_d435, include_top=include_top)
    print(f"[INFO] discovered jobs: {len(jobs)}")
    if not jobs:
        print("[INFO] no videos found. done.")
        return

    success_count = 0
    fail_count = 0
    for sequence_name, video_path in jobs:
        try:
            ok = run_single_job(
                video_path=video_path,
                sequence_name=sequence_name,
                server_host=args.server_host,
                server_port=args.server_port,
                server_user=args.server_user,
                server_alias=args.server_alias or None,
                remote_project_root=args.remote_project_root,
                remote_test_root=args.remote_test_root,
                remote_video_dir=args.remote_video_dir,
                local_output_dir=local_output_dir,
                conda_env=args.conda_env,
                use_vipe=args.use_vipe,
                gpu_id=args.gpu_id,
                dynhamr_fps=args.dynhamr_fps,
                sshpass_prefix=sshpass_prefix,
                skip_existing=args.skip_existing,
                download_full_results=args.download_full_results,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as error:
            fail_count += 1
            print(f"[FAIL] {sequence_name}: {error}")

    print("\n[SUMMARY]")
    print(f"success: {success_count}")
    print(f"failed:  {fail_count}")


if __name__ == "__main__":
    main()
