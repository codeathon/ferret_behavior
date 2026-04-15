# Tests

Unit and integration tests for the `bs` pipeline.

## Structure

```
tests/
├── conftest.py                 # Shared fixtures (reference geometry, kinematics factories, fake directories)
├── test_kinematics_core.py     # Quaternion, SLERP, RigidBodyKinematics, ReferenceGeometry
├── test_rigid_body_solver.py   # DLC metadata versioning, solver config, skull topology
├── test_ferret_gaze.py         # ClipPaths derivation, pipeline skip/reprocess logic
├── test_batch_processing.py    # full_pipeline overwrite flag cascade, step skipping
├── test_eye_analysis.py        # EyeVideoData models, CSV loading, session processing
├── test_utilities.py           # RecordingFolder construction, stage checks, PipelineStep enum
└── test_cameras.py             # Camera module: CameraProfile/CAMERAS, shared get_logger,
                                #   timestamp_utils, VideoWriterManager guards,
                                #   GrabLoopRunner pure logic

# Realtime-focused tests
├── test_realtime_transport.py      # packet schema, publisher factories, latency summary metrics
├── test_realtime_per_frame_compute.py # inference/triangulation backend contracts and factories
├── test_live_mocap_pipeline.py     # live tick/session orchestration (synthetic)
├── test_live_mocap_grab_session.py # grab session wiring into live publish path
├── test_grab_live_wiring.py        # queue policy, stage/publish error counters, consumer behavior
└── test_realtime_pipeline_e2e.py   # config + factories end-to-end wiring for live_mocap mode
```

## Setup

Install dev dependencies:

```bash
uv sync --group dev
```

Or install pytest directly:

```bash
uv add --dev pytest pytest-mock
```

## Running tests

### Run all tests

```bash
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

### Run a single test file

```bash
uv run pytest tests/test_cameras.py -v
uv run pytest tests/test_kinematics_core.py -v
```

### Run a single test class or function

```bash
uv run pytest tests/test_kinematics_core.py::TestQuaternion -v
uv run pytest tests/test_cameras.py::TestCAMERAS::test_all_known_serials_present -v
```

### Run with short traceback

```bash
uv run pytest --tb=short
```

### Stop on first failure

```bash
uv run pytest -x
```

### Run only tests matching a keyword

```bash
uv run pytest -k "slerp or quaternion"
uv run pytest -k "camera or timestamp"
```

## What is and is not tested

### Tested

| Module | What is covered |
|---|---|
| `kinematics_core` | Quaternion math, SLERP, normalization, rotation-matrix round-trip, RigidBodyKinematics shape/velocity/resample/save-load, ReferenceGeometry JSON round-trip |
| `rigid_body_solver` | DLC metadata version checks, RigidBodySolverConfig construction, skull/spine topology helpers |
| `ferret_gaze` | ClipPaths path derivation, eye-name assignment, output-existence checks, pipeline skip and reprocess flag logic |
| `batch_processing` | Overwrite flag cascade, auto-forcing DLC reprocess on outdated iteration, per-step skip when already done |
| `eye_analysis` | EyeType/EyeVideoData models, CSV load validation, function signature checks |
| `utilities` | RecordingFolder construction, missing directory errors, eye assignment, all is_*/check_* methods, PipelineStep enum |
| `cameras` | CameraProfile dataclass, CAMERAS single source of truth, derived constants (KNOWN_SERIALS, NO_BINNING_SERIALS), helper lookups, apply/configure camera settings, shared `get_logger` fallback, trim_timestamp_zeros, save_timestamps I/O, VideoWriterManager guard conditions, GrabLoopRunner seconds→frames math and drop-detection |
| `realtime` | transport schema/publisher contracts, per-frame inference/triangulation adapters, live mocap orchestration, grab-to-live queue wiring, resilience counters (`queue_overflow`, `stage_error`, `publish_error`) |

### Not tested

- Rerun and Blender visualization scripts (no return values to assert on)
- External subprocess tools (`skellyclicker`, `dlc_to_3d`, `freemocap`)
- Full numerical correctness of the Ceres solver (requires hardware)
- Live camera acquisition (actual pypylon grab loops, real ffmpeg pipes) — requires Basler hardware
- `src/old/` — archived code

## Adding new tests

1. Place test files in `tests/` following the `test_<module>.py` naming convention.
2. Add shared fixtures to `conftest.py`.
3. Use `tmp_path` (built-in pytest fixture) for any filesystem operations.
4. Mock external calls using `unittest.mock.patch` — never call real subprocesses in unit tests.
5. For hardware-dependent modules (cameras), patch `pypylon.pylon` in `sys.modules` before importing the module under test.
