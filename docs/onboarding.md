# Onboarding

This guide is for new contributors who want to run or modify the `bs` pipeline locally.

## 1) Prerequisites

- Python `3.12+` (repo is configured for 3.12)
- `uv` installed for environment management
- `ffmpeg` available on PATH (video processing)
- Optional but commonly needed:
  - GPU/CUDA-compatible setup for PyTorch workflows
  - Basler camera stack (`pypylon`) for acquisition workflows

## 2) Clone and environment setup

From repo root:

```bash
uv sync
uv run python --version
```

If dependency resolution fails, check:

- `pyproject.toml` for package constraints
- `uv.lock` for pinned state
- local CUDA/PyTorch compatibility if using GPU

## 3) Repository orientation

Core code directories:

- `python_code/cameras/`: recording, synchronization, camera postprocess
- `python_code/batch_processing/`: end-to-end session orchestration
- `python_code/ferret_gaze/`: gaze and resampling pipeline
- `python_code/kinematics_core/`: geometry and kinematic models
- `python_code/rigid_body_solver/`: rigid-body solver utilities
- `python_code/rerun_viewer/`: visual inspection and debug scripts

Docs and planning:

- `Writerside/`: deep technical docs map and topics
- `notes/`: internal notes and experiment planning

## 4) Common run paths

### A) Full session pipeline

Primary script: `python_code/batch_processing/full_pipeline.py`

What it orchestrates:

1. Video synchronization
2. Session calibration
3. DLC pose estimation
4. Triangulation
5. Eye/skull/gaze postprocessing

Run with:

```bash
uv run python python_code/batch_processing/full_pipeline.py
```

Important:

- The script currently includes machine-specific default paths in `__main__`.
- Before running, edit those defaults or call `full_pipeline(...)` from your own runner with explicit paths.
- This workflow also calls external tools/repos via subprocess (see section 6).

### B) Clip-level ferret gaze pipeline

Primary script: `python_code/ferret_gaze/run_gaze_pipeline.py`

What it does:

1. Builds eye kinematics from eye trajectories
2. Resamples skull/eye/toy/video data to shared timestamps
3. Computes gaze kinematics
4. Generates a Blender visualization script

Run with:

```bash
uv run python python_code/ferret_gaze/run_gaze_pipeline.py
```

Important:

- Update `recording_path` in `__main__` before running.
- The pipeline expects a specific input directory layout under `mocap_data/` and `eye_data/`.
- Outputs are written under `analyzable_output/` and `display_videos/`.

### C) Camera acquisition workflow

- Camera notes and operational flow are in `python_code/cameras/README.md`.
- Entry script is generally `python_code/cameras/multicamera_recording.py`.

## 5) Suggested first-day validation

Use this checklist after environment setup:

1. `uv run python -c "import cv2, pandas, torch; print('deps ok')"`
2. Open `python_code/ferret_gaze/run_gaze_pipeline.py` and inspect expected input paths.
3. Run one small test clip end-to-end.
4. Confirm `analyzable_output/gaze_kinematics/` files are generated.
5. Validate generated Blender script path and run it in Blender.

## 6) External dependencies to install separately

The batch pipeline references external projects in subprocess calls:

- `skellyclicker`
- `dlc_to_3d`
- `freemocap`

Before running full pipeline, verify:

- each repo is cloned locally
- scripts referenced in `full_pipeline.py` exist at your local paths
- Python environments used by those subprocesses are valid

## 7) Known onboarding footguns

- Absolute machine-specific paths in several scripts
- Hardware-dependent flows (Basler cameras, calibration assets)
- Sparse automated tests for the full end-to-end pipeline

Recommendation: create a small local wrapper script that passes all runtime paths from one config file instead of editing `__main__` blocks repeatedly.

## 8) Where to go deeper

- High-level docs tree: `Writerside/bs.tree`
- Camera workflow details: `python_code/cameras/README.md`
- Gaze pipeline code: `python_code/ferret_gaze/run_gaze_pipeline.py`
- Session pipeline code: `python_code/batch_processing/full_pipeline.py`
