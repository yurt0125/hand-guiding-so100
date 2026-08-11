# Project Framework (Integrated)

## Purpose
This repository is organized around a full hand-demonstration pipeline:
calibration -> RGB-D recording -> trajectory extraction -> trajectory validation -> robot execution.

## Directory Roles

### Runtime apps
- `app/`
  - `keyboard_mirror_app.py`: main real-robot + MuJoCo mirror keyboard runtime (keeps tuned control details).
  - `vision_app_refactored.py`: click-to-target visual runtime.
  - `replay_app_refactored.py`: CSV replay runtime.

### Shared runtime modules
- `common/`
  - `common_robot.py`: robot interface, kinematics wrapper, joint conversion.
  - `common_control.py`: smoothing/guard/joint filtering runner.
  - `config.py`: centralized runtime constants.
  - `model/`, `src/`: MuJoCo model assets and Pinocchio kinematics implementation.

### Calibration pipeline
- `calibration/`
  - `1_d435_click.py`: camera 3D point picking.
  - `2_handeye_calibration.py`: hand-eye transform estimation.
  - `points/`: future point-pair datasets.
  - `outputs/`: canonical calibration outputs (`cam2base_latest.json`).

### Offline pipelines
- `pipelines/record/1_d435_episode_record.py`
- `pipelines/extract/3_extract_traj.py`
- `pipelines/visualize/4_visualize_traj.py`
- `pipelines/export/` (reserved for VLA export scripts)

Compatibility wrappers are kept in `record/` to avoid breaking old command habits.

### Data and artifacts
- `data/raw_episodes/`: source-of-truth RGB-D episodes.
- `data/hamer_outputs/`: Dyn-HaMR outputs.
- `data/replay_csv/`: replay-ready CSVs.
- `data/trajectories/`: extracted trajectory tables.
- `data/vla_ready/`: reserved VLA training exports.
- `artifacts/previews/`: static images and visual snapshots.
- `artifacts/debug_videos/`: debug/export videos.

### Legacy
- `legacy/`: old scripts kept for reference only.

## Entrypoints
- `python run_pipeline.py --mode keyboard_mirror`
- `python run_pipeline.py --mode vision`
- `python run_pipeline.py --mode replay`

Legacy top-level wrappers remain:
- `2_control_ee_with_pinocchio_so100_real.py`
- `3_vision_grasp.py`
- `4_replay_csv_so100.py`
