# bs — Ferret Behavior Science Pipeline

End-to-end research pipeline for **ferret multi-camera recording, pose reconstruction, eye tracking, gaze analysis, and 3D visualization**. Built around Python 3.12, managed with `uv`, and targeting the Scholl Lab's lab workstation setup.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Module reference](#4-module-reference)
   - [kinematics_core](#kinematics_core)
   - [cameras](#cameras)
   - [video_viewing](#video_viewing)
   - [eye_analysis](#eye_analysis)
   - [rigid_body_solver](#rigid_body_solver)
   - [ferret_gaze](#ferret_gaze)
   - [rerun_viewer](#rerun_viewer)
   - [batch_processing](#batch_processing)
   - [utilities](#utilities)
5. [How to run each workflow](#5-how-to-run-each-workflow)
   - [Acquisition: multicamera recording](#acquisition-multicamera-recording)
   - [Acquisition: pupil eye recording](#acquisition-pupil-eye-recording)
   - [Full session pipeline](#full-session-pipeline)
   - [Clip-level gaze pipeline](#clip-level-gaze-pipeline)
   - [Inspect data with Rerun viewers](#inspect-data-with-rerun-viewers)
   - [Batch progress check](#batch-progress-check)
6. [Data directory structure](#6-data-directory-structure)
7. [External tools required](#7-external-tools-required)
8. [Dependency notes](#8-dependency-notes)
9. [Known issues and gotchas](#9-known-issues-and-gotchas)

---

## 1. What this project does

`bs` captures ferret behavior across synchronized Basler cameras and a Pupil eye tracker, then runs:

- **Camera synchronization** — aligns UTC timestamps across all camera streams.
- **Calibration** — multi-camera extrinsic capture volume calibration via Charuco board.
- **2D pose estimation** — DLC keypoints for head/body, eyes, and toy via `skellyclicker`.
- **3D triangulation** — multi-view reconstruction via `dlc_to_3d`.
- **Rigid-body skull solver** — fits a rigid body to noisy 3D marker trajectories using Ceres optimization to get clean skull pose + reference geometry.
- **Eye kinematics** — converts raw pupil/iris trajectories to eye rotation vectors in skull coordinates.
- **Gaze pipeline** — transforms eye rotations to world-frame gaze directions and produces analyzable kinematics.
- **Visualization** — Rerun 3D viewer and Blender scene generation for interactive inspection.

---

## 2. Repository layout

```
bs/
├── python_code/              # All Python source code
│   ├── kinematics_core/      # Shared data models (quaternions, rigid body, etc.)
│   ├── cameras/              # Basler camera acquisition and synchronization
│   ├── video_viewing/        # Video compositing and clip utilities
│   ├── eye_analysis/         # Eye video + DLC trajectory processing
│   ├── rigid_body_solver/    # Ceres-based rigid body fitting
│   ├── ferret_gaze/          # End-to-end gaze pipeline
│   ├── rerun_viewer/         # Rerun 3D visualization apps
│   ├── batch_processing/     # Session orchestration and batch runners
│   ├── utilities/            # Folder layout models, shared helpers
│   └── old/                  # Archived / experimental (not maintained)
├── Writerside/               # JetBrains Writerside knowledge-base docs
├── notes/                    # Research notes and planning
├── docs/                     # Project onboarding docs
├── pyproject.toml            # Python dependencies and uv config
└── uv.lock                   # Pinned lockfile
```

---

## 3. Setup

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — package and venv manager
- **ffmpeg** on PATH
- NVIDIA GPU + CUDA 12.1 for PyTorch GPU acceleration (optional but used in solver)
- Basler cameras + `pypylon` drivers for acquisition (lab workstation only)

### Install

```bash
git clone <repo-url>
cd bs
uv sync
```

### Verify

```bash
uv run python -c "import cv2, torch, pandas, rerun; print('OK')"
```

### PyTorch note

The project pins `torch` to the CUDA 12.1 index (`https://download.pytorch.org/whl/cu121`). If you are on CPU-only or a different CUDA version, edit `pyproject.toml` and rerun `uv sync`.

---

## 4. Module reference

### `kinematics_core`

**Path:** `python_code/kinematics_core/`

Foundational **Pydantic data models** for all time-series kinematic data. Every module that deals with poses or trajectories imports from here.

| File | Contents |
|---|---|
| `rigid_body_kinematics_model.py` | `RigidBodyKinematics` — positions + quaternions, velocity/acceleration helpers |
| `rigid_body_state_model.py` | `RigidBodyState` — single-frame pose |
| `quaternion_model.py` | `Quaternion`, SLERP, resampling utilities |
| `quaternion_trajectory_model.py` | `QuaternionTrajectory`, full trajectory SLERP helpers |
| `reference_geometry_model.py` | `ReferenceGeometry`, `CoordinateFrameDefinition`, `AxisDefinition`, `MarkerPosition` |
| `stick_figure_topology_model.py` | `StickFigureTopology` — edges and keypoints for Rerun/Blender rendering |
| `keypoint_trajectories.py` | `KeypointTrajectories` — named multi-point time series |
| `kinematics_serialization.py` | `save_kinematics`, `load_kinematics`, tidy CSV helpers |
| `derivative_helpers.py` | `compute_velocity`, `compute_angular_velocity` |

This module has **no entry points**; it is a library only.

---

### `cameras`

**Path:** `python_code/cameras/`

Basler multicamera acquisition via `pypylon`, UTC timestamp handling, Pupil synchronization, and session postprocessing.

| File | Role |
|---|---|
| `multicamera_recording.py` | Main acquisition script; `MultiCameraRecording` drives grab loops + ffmpeg writers |
| `postprocess.py` | `postprocess()` — runs sync, moves to `full_recording/`, calls `combine_videos` |
| `synchronization/timestamp_synchronize.py` | `TimestampSynchronize` — UTC alignment across cameras |
| `synchronization/timestamp_converter.py` | `TimestampConverter` — convert raw Basler ticks to UTC |
| `synchronization/pupil_synch.py` | Aligns Pupil timestamps to Basler |
| `intrinsics/intrinsics_calibration.py` | Per-camera lens intrinsics calibration |
| `intrinsics/intrinsics_corrector.py` | Apply distortion correction to frames |
| `diagnostics/` | Timestamp plotting and file exploration tools |

**Operational notes:** See `python_code/cameras/README.md` for step-by-step recording workflow. Recordings save to `/home/scholl-lab/recordings`.

---

### `video_viewing`

**Path:** `python_code/video_viewing/`

Composites multiple camera streams (Basler + Pupil) into single annotated video files; clips sessions; handles video rotation/flip layout.

| File | Role |
|---|---|
| `combine_videos.py` | Core compositing: `VideoInfo`, `combine_videos`, frame annotation |
| `combine_basler_videos.py` | Basler-specific combine wrapper |
| `clip_videos.py` | Clip Basler/Pupil streams with spatial transforms |
| `video_rotations.json` | Per-camera serial keys: position, rotation, layout position |
| `layout_locations.json` | Layout grid positions for the combined frame |
| `add_timestamps_to_dlc_output.py` | Append `timestamps_utc` column to DLC CSV |
| `closest_pupil_frame_to_basler_frame.py` | Frame-alignment helper for cross-stream queries |

---

### `eye_analysis`

**Path:** `python_code/eye_analysis/`

Processes raw DLC eye trajectory CSVs and eye videos: anatomical alignment between eyes, stabilized video export, and interactive Plotly dashboards.

| File | Role |
|---|---|
| `process_eye_session.py` | `process_eye_session_from_recording_folder()` — full session processing; called by `batch_processing` |
| `data_models/eye_video_dataset.py` | `EyeVideoData`, `EyeType` |
| `data_models/trajectory_dataset.py` | `Trajectory2D`, `ProcessedTrajectory`, `TrajectoryDataset` |
| `data_models/csv_io.py` | `load_trajectory_dataset` — DLC CSV loading and validation |
| `data_processing/align_data/eye_anatomical_alignment.py` | `eye_alignment_main` — spatial correction, merges per-eye CSVs |
| `data_processing/pupil_ellipse_fit.py` | Ellipse fitting for pupil boundary |
| `data_processing/active_contour_fit.py` | Active contour segmentation |
| `video_viewers/eye_viewer.py` | `EyeVideoDataViewer` — OpenCV viewer with overlays |
| `video_viewers/stabilized_eye_viewer.py` | `create_stabilized_eye_videos` — warped/stabilized eye video export |
| `video_viewers/image_overlay_system.py` | `OverlayRenderer` — draws elements on frames |
| `eye_plots/` | Plotly dashboards: timeseries, heatmaps, histogram, 3D surface, integrated layout |

**Entry points:**
```bash
uv run python python_code/eye_analysis/eye_analysis_main.py
uv run python python_code/eye_analysis/process_eye_session.py
```

---

### `rigid_body_solver`

**Path:** `python_code/rigid_body_solver/`

Fits a rigid body model to noisy 3D marker trajectories using **Ceres (`pyceres`) nonlinear optimization**. Produces clean skull pose, reference geometry (marker positions in body frame), and spine keypoints.

| File | Role |
|---|---|
| `core/main_solver_interface.py` | `RigidBodySolverConfig`, `process_tracking_data()` — full load → optimize → verify → save |
| `core/optimization.py` | `OptimizationConfig`, Ceres cost functions (`MeasurementFactor`, smoothness penalties), `optimize_rigid_body` |
| `core/calculate_reference_geometry.py` | `estimate_reference_geometry` — distance-matrix based body-frame reconstruction |
| `ferret_skull_solver.py` | `run_ferret_skull_solver_from_recording_folder()` — ferret-specific entry; skull + spine topology |
| `data_io/load_measured_trajectories.py` | `load_measured_trajectories_csv` |
| `viz/ferret_skull_rerun.py` | Log skull/spine, gaze vectors to Rerun |
| `viz/ferret_skull_blender_viz.py` | Build Blender scene with skull bones and kinematics |
| `viz/plot_trajectories.py` | Matplotlib trajectory debug plots |

**Outputs** are written to `solver_output/` inside the recording folder:
```
solver_output/
├── skull_kinematics.csv
├── skull_reference_geometry.json
├── skull_and_spine_trajectories.csv
└── skull_and_spine_topology.json
```

**Entry point:**
```bash
uv run python python_code/rigid_body_solver/ferret_skull_solver.py
```

---

### `ferret_gaze`

**Path:** `python_code/ferret_gaze/`

The central analysis module. Orchestrates eye kinematics → resampling → gaze computation → visualization. Four stages, each idempotent (skips if output exists).

#### Submodules

**`eye_kinematics/`** — converts raw DLC pupil/iris trajectories to 3D eye rotation vectors in skull coordinates.

| File | Role |
|---|---|
| `ferret_eye_kinematics_models.py` | `FerretEyeKinematics`, `SocketLandmarks` |
| `ferret_eye_kinematics_functions.py` | `eye_camera_distance_from_skull_geometry`, loading helpers |
| `ferret_eyeball_reference_geometry.py` | Anatomical constants for the ferret eyeball |
| `torsion_estimation.py` | Torsional rotation estimation |
| `run_eye_kinematics_pipeline.py` | Standalone entry point for eye kinematics only |
| `eye_kinematics_rerun_viewer.py` | Rerun logging helpers; also provides shared color constants used across modules |

**`data_resampling/`** — resamples all streams (skull, eye, toy, videos) to a shared common timestamp array.

| File | Role |
|---|---|
| `ferret_data_resampler.py` | `resample_ferret_data()`, `VideoConfig` |
| `video_resampler.py` | Per-video resampling using timestamp alignment |
| `data_resampling_helpers.py` | `ResamplingStrategy` enum, interpolation helpers |
| `toy_trajectory_loader.py` | Load toy body 3D CSV |

**`calculate_gaze/`** — transforms resampled skull + eye rotations into world-frame gaze kinematics.

| File | Role |
|---|---|
| `calculate_ferret_gaze.py` | `calculate_ferret_gaze()` — coordinate-frame-documented gaze computation |
| `ferret_gaze_kinematics.py` | `FerretGazeKinematics` model |
| `synthetic_gaze_test.py` | Synthetic data validation tests |

**`visualization/`** — Rerun + Blender + matplotlib figure generation.

| File | Role |
|---|---|
| `ferret_gaze_rerun.py` | Head + gaze Rerun scene |
| `ferret_gaze_blender/ferret_full_gaze_blender_viz.py` | Full Blender scene: head, eyes, gaze vectors, toy, video planes |
| `plot_ferret_kinematics_vor.py` | VOR (vestibulo-ocular reflex) analysis plots |
| `plot_vor_correlation_grid.py` | VOR correlation grid figure |
| `plot_head_yaw_vs_eye_horizontal_velocity.py` | Coordination analysis plot |
| `gaze_line_plots/` | Time-series gaze line figures |

**Top-level orchestrator:**

| File | Role |
|---|---|
| `run_gaze_pipeline.py` | `run_gaze_pipeline(recording_path, ...)` — runs all 4 steps; each step is skipped if outputs exist |

**Entry point:**
```bash
uv run python python_code/ferret_gaze/run_gaze_pipeline.py
```

**Reprocess flags** let you force specific stages:
```python
run_gaze_pipeline(
    recording_path=Path("..."),
    reprocess_eye_kinematics=True,  # re-run step 1
    reprocess_gaze=True,            # re-run step 3
    reprocess_all=True,             # re-run everything
)
```

---

### `rerun_viewer`

**Path:** `python_code/rerun_viewer/`

Collection of **Rerun** viewer apps for interactive inspection of raw and processed data. Each viewer targets a different data type or pipeline stage.

| Script | What it shows |
|---|---|
| `everything_viewer.py` | Composite: videos, 3D markers, gaze, eye traces |
| `3d_data_viewer.py` | 3D solver output / triangulated keypoints |
| `3d_data_comparison.py` | Side-by-side 3D data comparison |
| `all_video_viewer.py` | All synchronized camera streams |
| `eyes_and_head_rotation.py` | Eye traces + head rotation timeseries + video |
| `charuco_3d_viewer.py` | Calibration Charuco board 3D view |
| `rerun_clip_view_creator.py` | Create a Rerun session for a specific clip |
| `plot_resampled_data.py` | Resampled analyzable output plots |

**`rerun_utils/`** — shared building blocks used by all viewer scripts.

| File | Role |
|---|---|
| `recording_folder.py` | Viewer-specific `RecordingFolder` with paths to all data assets |
| `video_data.py` | `MocapVideoData`, `EyeVideoData`, `WorldCameraVideoData`, etc. |
| `process_videos.py` | Frame-by-frame processing for Rerun logging |
| `gaze_plots/` | Skull/spine, gaze, naive gaze, 3D eye — logs to Rerun 3D |
| `plot_3d_data.py`, `plot_eye_traces.py`, `plot_head_rotation.py`, etc. | Rerun entity loggers |
| `groundplane_and_origin.py`, `log_cameras.py` | Scene setup |

**Example:**
```bash
uv run python python_code/rerun_viewer/everything_viewer.py
uv run python python_code/rerun_viewer/eyes_and_head_rotation.py
```

---

### `batch_processing`

**Path:** `python_code/batch_processing/`

High-level orchestration scripts for processing one session or many sessions.

| File | Role |
|---|---|
| `full_pipeline.py` | `full_pipeline()` — sync → calibrate → DLC → triangulate → postprocess; calls external repos via subprocess |
| `postprocess_recording.py` | `process_recording()` — eye analysis + skull solver + gaze; runs after triangulation |
| `batch_synchronize.py` | Loop all sessions, run `cameras.postprocess.postprocess` on each |
| `check_progress/check_progress.py` | `check_progress(ferret_recordings_path)` → DataFrame of pipeline stage flags across all sessions |

**Entry points:**
```bash
uv run python python_code/batch_processing/full_pipeline.py
uv run python python_code/batch_processing/postprocess_recording.py
uv run python python_code/batch_processing/check_progress/check_progress.py
```

**DLC iteration pinning:** The full pipeline tracks `skellyclicker_metadata.json` iteration numbers to automatically force DLC reprocessing when models are updated:

```python
HEAD_DLC_ITERATION = 17
EYE_DLC_ITERATION  = 30
TOY_DLC_ITERATION  = 10
```

---

### `utilities`

**Path:** `python_code/utilities/`

Shared path models and pipeline validation helpers used across the entire codebase.

| File | Role |
|---|---|
| `folder_utilities/recording_folder.py` | `RecordingFolder` — canonical session path model; `is_synchronized()`, `is_calibrated()`, `is_dlc_processed()`, `is_triangulated()`, `is_gaze_postprocessed()`, etc. |
| `folder_utilities/calibration_folder.py` | `CalibrationFolder` — calibration session layout |
| `folder_utilities/top_level_folder.py` | `TopLevelFolder` — root recordings directory |
| `connections_and_landmarks.py` | `ferret_head_spine_landmarks`, `toy_landmarks`, connection indices for Rerun 3D plots |
| `get_mean_dlc_confidence.py` | DLC quality scoring |
| `find_bad_eye_data.py` | Flag sessions with bad eye tracking data |
| `tidy_machine_labels.py` | Normalize DLC label names to match timestamps |
| `clip_any_videos.py` | Standalone video clipping utility |

---

## 5. How to run each workflow

### Acquisition: multicamera recording

1. Open `python_code/cameras/multicamera_recording.py` in VSCode/Cursor.
2. Edit the `recording_name` variable at the bottom of the file.
3. Uncomment exactly **one** grab mode:
   - `grab_n_frames(n)` — record exactly N frames
   - `grab_n_seconds(t)` — record for t seconds (e.g. `20*60`)
   - `grab_until_input()` — record until you press Enter
4. Run the file. Output saves to `/home/scholl-lab/recordings/<session>/`.

> If cameras have been moved or it is the first recording of the day, recalibrate first.

---

### Acquisition: pupil eye recording

```bash
conda activate pupil_source
cd Documents/git_repos/pupil/pupil_src
python main.py capture
```

- Set `Recorder: Recording session name` to match your Basler session name.
- Press `R` to start / stop recording.
- Saved to `/home/scholl-lab/pupil_recordings`.

---

### Full session pipeline

Runs synchronization → calibration → DLC → triangulation → eye/skull/gaze postprocessing in one call.

Edit `__main__` in `full_pipeline.py` with your session path, then:

```bash
uv run python python_code/batch_processing/full_pipeline.py
```

Use boolean flags to skip completed stages and force specific rewrites:

```python
full_pipeline(
    recording_folder_path=Path("/home/scholl-lab/ferret_recordings/.../full_recording"),
    overwrite_synchronization=False,
    overwrite_calibration=False,
    overwrite_dlc=False,
    overwrite_triangulation=False,
    overwrite_eye_postprocessing=True,
    overwrite_skull_postprocessing=False,
    overwrite_gaze=True,
)
```

---

### Clip-level gaze pipeline

Run after triangulation + solver output exist for a clip directory.

Edit `__main__` in `run_gaze_pipeline.py` with your clip path, then:

```bash
uv run python python_code/ferret_gaze/run_gaze_pipeline.py
```

Or from Python:

```python
from python_code.ferret_gaze.run_gaze_pipeline import run_gaze_pipeline
from pathlib import Path

run_gaze_pipeline(
    recording_path=Path("/home/scholl-lab/ferret_recordings/.../clips/0m_37s-1m_37s"),
    reprocess_all=True,
)
```

**Required inputs** (relative to `clip_path`):
```
mocap_data/output_data/solver_output/skull_kinematics.csv
mocap_data/output_data/solver_output/skull_reference_geometry.json
eye_data/output_data/eye0_data.csv
eye_data/output_data/eye1_data.csv
```

**Output locations:**
```
analyzable_output/gaze_kinematics/left_gaze_kinematics.csv
analyzable_output/gaze_kinematics/right_gaze_kinematics.csv
display_videos/top_down_mocap_resampled.mp4
display_videos/left_eye_resampled.mp4
analyzable_output/ferret_full_gaze_blender_viz.py  ← run this in Blender
```

---

### Inspect data with Rerun viewers

After any processing stage, launch a viewer:

```bash
# Everything at once
uv run python python_code/rerun_viewer/everything_viewer.py

# Eye traces + head rotation
uv run python python_code/rerun_viewer/eyes_and_head_rotation.py

# 3D solver output
uv run python python_code/rerun_viewer/3d_data_viewer.py
```

All viewers open an interactive Rerun session in the browser or Rerun desktop app.

---

### Blender visualization

After the gaze pipeline runs:

1. Open Blender 4.0+.
2. Open the generated script at `analyzable_output/ferret_full_gaze_blender_viz.py`.
3. Run it with `Alt+P`.
4. Press `Spacebar` to play the animation.

---

### Batch progress check

```bash
uv run python python_code/batch_processing/check_progress/check_progress.py
```

Returns a DataFrame with per-session pipeline stage completion flags and DLC iteration numbers.

---

## 6. Data directory structure

### Session layout (under `/home/scholl-lab/ferret_recordings/`)

```
session_YYYY-MM-DD_ferret_<id>_<tag>/
├── calibration_videos/          # Charuco board recordings
├── full_recording/
│   ├── mocap_data/
│   │   ├── synchronized_corrected_videos/   # aligned per-cam .mp4
│   │   ├── synchronized_timestamps/
│   │   └── output_data/
│   │       ├── dlc/
│   │       │   ├── head_body_3d_xyz.csv
│   │       │   └── toy_body_3d_xyz.csv
│   │       └── solver_output/
│   │           ├── skull_kinematics.csv
│   │           ├── skull_reference_geometry.json
│   │           └── skull_and_spine_trajectories.csv
│   └── eye_data/
│       ├── eye_videos/
│       ├── output_data/
│       │   ├── eye0_data.csv
│       │   └── eye1_data.csv
│       ├── left_eye_stabilized.mp4
│       └── right_eye_stabilized.mp4
└── clips/
    └── <clip_name>/
        ├── mocap_data/          # (same layout as full_recording)
        ├── eye_data/            # (same layout as full_recording)
        ├── analyzable_output/
        │   ├── common_timestamps.npy
        │   ├── skull_kinematics/
        │   ├── left_eye_kinematics/
        │   ├── right_eye_kinematics/
        │   ├── gaze_kinematics/
        │   └── ferret_full_gaze_blender_viz.py
        └── display_videos/
```

---

## 7. External tools required

`full_pipeline.py` calls three external repos via subprocess. Clone and set up separately:

| Tool | Repo | Used for |
|---|---|---|
| `skellyclicker` | `github.com/freemocap/skellyclicker` | DLC pose estimation |
| `dlc_to_3d` | `github.com/philipqueen/freemocap_playground@philip/bs` | Multi-view 3D triangulation |
| `freemocap` | `github.com/freemocap/freemocap` | Headless calibration |

Default subprocess paths in `full_pipeline.py` point to `/home/scholl-lab/...`. Update them to match your local install if running on a different machine.

---

## 8. Dependency notes

- **`pypylon`** is listed twice in `pyproject.toml` (pinned `==4.2.0` and `>=4.2.0`) — harmless but worth cleaning up.
- **`torch`** is pulled from the PyTorch CUDA 12.1 index (`pytorch-cu121`). CPU-only machines need to override this.
- **`pyceres`** requires a local or pip-installed Ceres build — verify this installs cleanly before running the solver.
- **`jaxlib`** is declared but not actively imported in current code paths — likely a future or optional dependency.

---

## 9. Known issues and gotchas

| Issue | Details |
|---|---|
| Hardcoded paths in `__main__` blocks | Most scripts have machine-specific paths hardcoded. Always update before running. |
| `cube_solver_demo.py` is stale | Imports `TrackingConfig` and `data_io.data_savers` which no longer exist in the current API. |
| Left/right eye assignment is ferret-specific | `ClipPaths.left_eye_name` in `run_gaze_pipeline.py` flips based on whether `"757"` is in the session path. |
| `eye_analysis_main.py` has Windows paths | Hard-coded drive paths from development; replace before running on Linux/Mac. |
| No pytest CI | Tests are ad hoc scripts. There is no `pytest` config and no CI test gate. |
| `old/` directory | Contains archived code with stale imports. Do not import from `python_code.old` in new code. |

---

## Further reading

- Camera operational guide: `python_code/cameras/README.md`
- Gaze pipeline narrative: `python_code/ferret_gaze/gaze_pipeline_explanation.md`
- Knowledge base: `Writerside/` (open with JetBrains Writerside or build to HTML)
- Onboarding guide: `docs/onboarding.md`
