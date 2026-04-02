# Project Design Document
## bs — Ferret Behavior Science Pipeline

> **Purpose of this document:** A reusable design reference that captures the architecture, conventions, and patterns of this project. Feed it to an AI coding assistant at the start of a new session or a new project to give it full context.

---

## 1. Project overview

**What it is:** An end-to-end research pipeline for multi-camera animal behavior recording, 3D pose reconstruction, eye tracking, gaze analysis, and visualization.

**Language / runtime:** Python 3.12, managed with `uv` (lockfile-based, no conda).

**Scale:** ~236 source files, ~515 git commits, 9 active modules.

**Scientific goal:** Record a freely-moving ferret with synchronized Basler cameras + Pupil eye tracker, reconstruct skull and eye poses in 3D, and compute where each eye is looking in the world at every frame.

---

## 2. Repository layout

```
<project>/
├── src/               # All source code — one sub-package per domain
├── tests/             # pytest unit tests — test_<module>.py per package
├── docs/              # Onboarding guide and design documents
├── notes/             # Research notes (not code)
├── Writerside/        # JetBrains Writerside knowledge-base (optional)
├── pyproject.toml     # Dependencies, pytest config, uv config
└── uv.lock            # Pinned lockfile
```

**Rule:** Source code lives in `src/`, test code lives in `tests/`. No exceptions.

---

## 3. Module architecture

### Layer model

The project is structured as explicit layers — lower layers never import from higher layers.

```
┌─────────────────────────────────────────────┐
│  batch_processing/   (orchestration)        │  ← top: runs everything
├─────────────────────────────────────────────┤
│  ferret_gaze/        (domain pipeline)      │
│  eye_analysis/       (domain pipeline)      │
│  rigid_body_solver/  (domain pipeline)      │
├─────────────────────────────────────────────┤
│  cameras/            (hardware + sync)      │
│  rerun_viewer/       (visualization)        │
│  video_viewing/      (video assembly)       │
├─────────────────────────────────────────────┤
│  kinematics_core/    (math library)         │  ← shared, no side effects
│  utilities/          (path models + I/O)    │  ← shared, no domain logic
└─────────────────────────────────────────────┘
```

### Module responsibilities

| Module | Responsibility | Entry point |
|---|---|---|
| `kinematics_core/` | Pydantic data models for 3D rigid body kinematics (poses, quaternions, trajectories, serialization). No domain logic, no I/O side effects. | None — library only |
| `utilities/` | `RecordingFolder` and `CalibrationFolder` path models, pipeline-stage validation, shared logging config, shared Polars tidy helpers | None — library only |
| `cameras/` | Basler multicamera acquisition, timestamp sync, Pupil alignment, intrinsics calibration, session postprocessing | `run_recording.py` |
| `eye_analysis/` | DLC eye trajectory loading, anatomical alignment, signal processing, stabilized video export | `eye_analysis_main.py` |
| `rigid_body_solver/` | Ceres-based rigid body fitting to 3D markers; outputs skull kinematics JSON | `ferret_skull_solver.py` |
| `ferret_gaze/` | Eye kinematics → timestamp resampling → world-frame gaze → visualization | `run_gaze_pipeline.py` |
| `video_viewing/` | Multi-camera video assembly, clipping, layout config | Scripts |
| `rerun_viewer/` | Rerun 3D viewer apps for any pipeline stage | Scripts |
| `batch_processing/` | Session-level orchestration of all pipeline stages | `full_pipeline.py` |

---

## 4. Core design patterns

### 4.1 Pydantic for all data models

All data models use `pydantic.BaseModel` with:
- `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)` for immutable trajectory objects
- Validators (`@model_validator`) for shape / consistency checks
- `@classmethod` factory methods named `from_<source>` for construction

**Example pattern:**
```python
class RigidBodyKinematics(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    timestamps: NDArray[np.float64]      # (N,)
    positions_xyz: NDArray[np.float64]   # (N, 3) in mm

    @model_validator(mode="after")
    def validate_shapes(self) -> "RigidBodyKinematics":
        ...

    @classmethod
    def from_csv(cls, path: Path) -> "RigidBodyKinematics":
        ...
```

### 4.2 Tidy serialization format

All kinematics data serializes to a **tidy CSV** (one row per observation):

| frame | timestamp_s | trajectory | component | value | units |
|---|---|---|---|---|---|
| 0 | 0.000 | position | x | 12.3 | mm |
| 0 | 0.000 | position | y | -4.1 | mm |

Plus a companion **reference geometry JSON** for named landmark positions.

Helper: `src/utilities/polars_tidy.py` → `build_vector_chunk()` and `extract_timestamps_from_tidy_df()`.

### 4.3 Path models for filesystem contracts

All pipeline stages use typed path models (`RecordingFolder`, `CalibrationFolder`) that:
- Validate directory existence and expected structure
- Provide named `@property` accessors for every expected file/directory
- Expose `is_<stage>()` and `check_<stage>()` methods for pipeline gating

**Rule:** Never use bare `Path("some/hardcoded/path")` strings in pipeline code. Use a path model.

### 4.4 Idempotent pipeline stages

Every stage checks whether its output already exists before running. Controlled by `overwrite_<stage>: bool = False` flags, cascading from the orchestrator:

```python
def full_pipeline(
    session_path: Path,
    overwrite_sync: bool = False,
    overwrite_calibration: bool = False,
    overwrite_dlc: bool = False,
    ...
):
```

### 4.5 Shared logging via `src/utilities/logging_config.py`

**Rule:** Never use `import logging` + `logging.getLogger()` directly. Always:
```python
from src.utilities.logging_config import get_logger
logger = get_logger(__name__)
```

The shared `get_logger` configures a root logger `"bs"` once (console + optional file handler with timestamped filename) and returns named child loggers for all modules.

### 4.6 External math libraries via adapter module

For optional backends (SciPy, numpy-quaternion, pytransform3d), use the adapter pattern in `src/kinematics_core/math_backends.py`:

- Pydantic models own the data in project conventions (`[w, x, y, z]` quaternions, mm units)
- Adapters handle convention conversion at compute boundaries
- Backends are optional: `ImportError` triggers graceful fallback to pure-NumPy implementations
- Never let library-specific types leak into Pydantic models or public APIs

---

## 5. Quaternion convention

**Project convention: `[w, x, y, z]` scalar-first.**

SciPy uses `[x, y, z, w]`. Use `math_backends.wxyz_to_xyzw()` and `xyzw_to_wxyz()` at all boundaries.

Key operations:
- Composition: `q_result = q_a * q_b` (Hamilton product, `q_b` applied first)
- Rotation: `v' = q * v * q⁻¹` (use `rotate_vector` method)
- Interpolation: SLERP via `QuaternionTrajectory.resample()` (SciPy if available, NumPy fallback)
- Storage: `(N, 4)` float64 NumPy array in all trajectory models

---

## 6. Timestamp convention

All timestamps are **UTC nanoseconds** (`int`) from `time.time_ns()` at recording time.

Conversions:
- `seconds_to_nanoseconds(s)` → `int(s * 1e9)`
- `nanoseconds_to_seconds(ns)` → `ns / 1e9`

Both available in `src/cameras/synchronization/time_units.py`.

Synchronized video directories resolve via `resolve_synchronized_video_dir(folder)` (prefers `synchronized_corrected_videos`, falls back to `synchronized_videos`).

---

## 7. Data directory structure

```
session_YYYY-MM-DD_<subject>/
├── calibration_videos/
├── full_recording/
│   ├── mocap_data/
│   │   ├── raw_videos/                 # Basler .mp4 + timestamps.npy
│   │   ├── synchronized_corrected_videos/
│   │   └── output_data/
│   │       ├── dlc/                    # DLC CSVs
│   │       └── solver_output/          # skull_kinematics.csv + JSON
│   └── eye_data/
│       ├── eye_videos/
│       └── output_data/
│           └── eye_kinematics/         # per-eye kinematics CSV + JSON
└── clips/<clip_name>/
    ├── mocap_data/
    ├── eye_data/
    ├── analyzable_output/              # gaze CSVs, Blender script
    └── display_videos/                 # resampled .mp4 files
```

---

## 8. Testing conventions

- All tests in `tests/`, named `test_<module>.py`
- Run with `uv run pytest`
- Shared fixtures in `tests/conftest.py` (`tmp_path`, fake folder structures, kinematics factories)
- **Never call real subprocesses, real cameras, or real ffmpeg in tests** — mock everything hardware-dependent
- For camera tests: patch `pypylon.pylon` in `sys.modules` before importing anything from `src/cameras/`
- Hardware-dependent modules (cameras, Ceres solver) are tested via mocks at the class boundary

---

## 9. Git workflow

- **Branch naming:** `feature/`, `fix/`, `refactor/`, `docs/`
- **One logical change per commit** with conventional commit messages: `feat:`, `fix:`, `refactor:`, `docs:`
- **Scan dependencies** before committing: run `grep -r "from src.<module>"` to check all importers of a changed file
- **Never push directly to `main`** without a branch + merge
- **Remote:** `ferret` → `https://github.com/codeathon/ferret_behavior.git`

---

## 10. Code style rules

- Functions ≤ 45 lines
- Comments explain *why*, not *what*
- No `print()` anywhere — always `logger.info/debug/warning/error`
- Source in `src/`, tests in `tests/` — never mix
- Use format strings in logger calls: `logger.info("value: %s", val)` not f-strings (avoids string formatting when log level is suppressed)
- `Path.exists()` not `Path.exists` (missing `()` is a silent always-true bug)

---

## 11. Dependency management

```toml
[project]
requires-python = ">=3.12"

[tool.uv]
index-strategy = "unsafe-best-match"

[[tool.uv.index]]
name = "pytorch-cu121"
url  = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
```

Add dependencies: `uv add <package>`. Dev-only: `uv add --dev <package>`.

Optional backends (`scipy`, `numpy-quaternion`, `pytransform3d`) are not in `pyproject.toml` — imported with `try/except ImportError` so the core pipeline runs without them.

---

## 12. Key shared utilities quick reference

| What you need | Where to find it |
|---|---|
| Logger for any module | `from src.utilities.logging_config import get_logger` |
| Tidy DataFrame chunk builder | `from src.utilities.polars_tidy import build_vector_chunk` |
| Recording folder path model | `from src.utilities.folder_utilities.recording_folder import RecordingFolder` |
| Calibration folder path model | `from src.utilities.folder_utilities.calibration_folder import CalibrationFolder` |
| Quaternion convention adapters | `from src.kinematics_core.math_backends import scipy_rotation_from_wxyz, wxyz_from_scipy_rotation` |
| ns ↔ s conversion | `from src.cameras.synchronization.time_units import seconds_to_nanoseconds, nanoseconds_to_seconds` |
| Sync video dir resolution | `from src.cameras.synchronization.time_units import resolve_synchronized_video_dir` |

---

## 13. Known technical debt

| Item | Location | Risk |
|---|---|---|
| Hardcoded lab paths in `__main__` blocks | Most pipeline scripts | Medium — update before running on new machines |
| `video_resampler.py` unused (superseded by `ferret_data_resampler.py`) | `src/ferret_gaze/data_resampling/` | Low — candidate for deletion |
| `combine_videos.py` vs `combine_basler_videos.py` duplication | `src/video_viewing/` | High to merge — keep separate for now |
| No CI gate — tests run manually | — | Medium |
| `left/right` eye assignment keyed to ferret ID `"757"` in path string | `run_gaze_pipeline.py` | Medium — needs config |
| `src/old/` contains bare `import logging` throughout | `src/old/` | Low — archived, do not import |
