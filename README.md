# bs — Ferret Behavior Science Pipeline

End-to-end research pipeline for ferret multi-camera recording, pose reconstruction, eye tracking, gaze analysis, and 3D visualization. Built on Python 3.12, managed with `uv`.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Module reference](#4-module-reference)
5. [How to run each workflow](#5-how-to-run-each-workflow)
6. [Data directory structure](#6-data-directory-structure)
7. [External tools required](#7-external-tools-required)
8. [Known issues](#8-known-issues)

---

## 1. What this project does

- **Camera synchronization** — aligns UTC timestamps across all Basler + Pupil camera streams.
- **Calibration** — multi-camera extrinsic calibration via Charuco board.
- **2D pose estimation** — DLC keypoints for head/body, eyes, and toy via `skellyclicker`.
- **3D triangulation** — multi-view reconstruction via `dlc_to_3d`.
- **Rigid-body skull solver** — Ceres optimization to recover clean skull pose and reference geometry from noisy markers.
- **Eye kinematics** — raw pupil/iris trajectories → eye rotation vectors in skull coordinates.
- **Gaze pipeline** — eye rotations → world-frame gaze directions and analyzable kinematics.
- **Visualization** — Rerun 3D viewer and Blender scene generation.

---

## 2. Repository layout

```
bs/
├── src/               # All source code (see Module reference)
├── tests/             # pytest unit test suite (cameras, kinematics, gaze, batch)
├── Writerside/        # JetBrains Writerside knowledge-base docs
├── notes/             # Research notes and planning
├── docs/              # Onboarding guide
├── pyproject.toml     # Dependencies and uv config
└── uv.lock            # Pinned lockfile
```

---

## 3. Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), ffmpeg on PATH, CUDA 12.1 for GPU (optional).

```bash
git clone <repo-url> && cd bs
uv sync
uv run python -c "import cv2, torch, pandas, rerun; print('OK')"
```

> `torch` is pinned to the CUDA 12.1 PyTorch index. For CPU-only machines, remove the `[tool.uv.sources]` block from `pyproject.toml` before syncing.

---

## 4. Module reference

| Module | Purpose |
|---|---|
| `src/kinematics_core/` | Shared Pydantic data models for poses, quaternions, trajectories, and reference geometry. Library only — no entry points. |
| `src/cameras/` | Basler multicamera acquisition, UTC timestamp sync, Pupil alignment, intrinsics calibration, and session postprocessing. Internally split into focused modules: `camera_config.py` (hardware config), `video_writers.py` (ffmpeg/OpenCV writers), `timestamp_utils.py` (latch/save), `grab_loops.py` (frame loop). Run a session via `run_recording.py`. |
| `src/video_viewing/` | Composite multi-camera video assembly, session clipping, and rotation/flip layout config. |
| `src/eye_analysis/` | DLC eye trajectory loading, anatomical alignment, stabilized video export, and Plotly dashboards. |
| `src/rigid_body_solver/` | Ceres-based rigid body fitting to 3D markers; outputs skull kinematics and reference geometry JSON. |
| `src/ferret_gaze/` | Full gaze pipeline: eye kinematics → resampling → gaze computation → Rerun/Blender visualization. Each stage is idempotent. |
| `src/rerun_viewer/` | Rerun viewer apps for interactive inspection of videos, 3D markers, gaze, and eye traces at any pipeline stage. |
| `src/batch_processing/` | Session-level orchestration: runs sync, calibration, DLC, triangulation, and postprocessing in sequence; batch progress dashboard. |
| `src/utilities/` | `RecordingFolder` path model and pipeline-stage validation helpers used across the whole codebase. |
| `src/old/` | Archived/experimental code. Do not import from here. |

---

## 5. How to run each workflow

### Multicamera recording
Edit the `CONFIG` section at the top of `run_recording.py` (set `RECORDING_NAME`, `FPS`, `BINNING_FACTOR`, etc.), uncomment one grab mode, then run:
```bash
uv run python src/cameras/run_recording.py
```
See `src/cameras/README.md` for full operational steps.

### Pupil eye recording
```bash
conda activate pupil_source && cd Documents/git_repos/pupil/pupil_src
python main.py capture
```

### Full session pipeline
Edit the path in `__main__` of `full_pipeline.py`, then:
```bash
uv run python src/batch_processing/full_pipeline.py
```
Pass `overwrite_*` booleans to re-run specific stages (sync, calibration, DLC, triangulation, eye/skull/gaze postprocessing).

### Realtime scaffold pipeline (JSON config)
Use the checked-in config scaffold at `configs/realtime.runtime.json` and call `run_pipeline(..., mode="realtime", realtime_config_path=...)` from Python:
```bash
uv run python -c "from pathlib import Path; from src.batch_processing.full_pipeline import run_pipeline; run_pipeline(recording_folder_path=Path('/tmp/full_recording'), mode='realtime', realtime_config_path=Path('configs/realtime.runtime.json'))"
```

### Clip-level gaze pipeline
Edit the clip path in `__main__` of `run_gaze_pipeline.py`, then:
```bash
uv run python src/ferret_gaze/run_gaze_pipeline.py
```
Pass `reprocess_all=True` or individual `reprocess_*` flags to force specific stages.

### Rerun viewers
```bash
uv run python src/rerun_viewer/everything_viewer.py
uv run python src/rerun_viewer/eyes_and_head_rotation.py
```

### Blender visualization
After the gaze pipeline runs, open `analyzable_output/ferret_full_gaze_blender_viz.py` in Blender 4.0+, run with `Alt+P`, press `Spacebar` to play.

### Batch progress check
```bash
uv run python src/batch_processing/check_progress/check_progress.py
```

### Running the test suite
Install dev dependencies then run all tests:
```bash
uv sync --group dev
uv run pytest
```

Run a specific module's tests:
```bash
uv run pytest tests/test_cameras.py -v
uv run pytest tests/test_kinematics_core.py -v
```

Stop on first failure with short tracebacks:
```bash
uv run pytest -x --tb=short
```

See `tests/README.md` for the full test inventory and how to add new tests.

---

## 6. Data directory structure

```
session_YYYY-MM-DD_ferret_<id>/
├── calibration_videos/
├── full_recording/
│   ├── mocap_data/
│   │   ├── synchronized_corrected_videos/
│   │   └── output_data/
│   │       ├── dlc/
│   │       └── solver_output/
│   └── eye_data/
│       ├── eye_videos/
│       └── output_data/
└── clips/<clip_name>/
    ├── mocap_data/
    ├── eye_data/
    ├── analyzable_output/     # gaze kinematics, Blender script
    └── display_videos/        # resampled mp4s
```

---

## 7. External tools required

`full_pipeline.py` calls these via subprocess — clone and install separately:

| Tool | Repo | Used for |
|---|---|---|
| `skellyclicker` | `github.com/freemocap/skellyclicker` | DLC pose estimation |
| `dlc_to_3d` | `github.com/philipqueen/freemocap_playground@philip/bs` | 3D triangulation |
| `freemocap` | `github.com/freemocap/freemocap` | Headless calibration |

Subprocess paths in `full_pipeline.py` default to `/home/scholl-lab/...` — update for your machine.

---

## 8. Known issues

- Most `__main__` blocks contain hardcoded lab machine paths — update before running. Exception: `src/cameras/run_recording.py` has a dedicated `CONFIG` section at the top for this.
- `cube_solver_demo.py` imports stale APIs that no longer exist.
- Left/right eye assignment in `run_gaze_pipeline.py` is keyed to ferret ID `"757"` in the path string.
- No CI test gate exists — tests must be run manually via `uv run pytest`.

---

Further reading: `docs/onboarding.md` · `src/cameras/README.md` · `Writerside/`
