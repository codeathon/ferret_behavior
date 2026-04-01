# Cameras

Basler multi-camera acquisition, UTC timestamp synchronization, Pupil eye tracking alignment, and session postprocessing.

## Module structure

After the `camera_restructure` refactor, the acquisition code is split across focused files:

| File | Responsibility |
|---|---|
| `run_recording.py` | **Operator entry point.** Edit the `CONFIG` block at the top, then run this file. |
| `multicamera_recording.py` | `MultiCameraRecording` orchestrator class. Wires everything together; public API unchanged. |
| `camera_config.py` | `ImageShape`, per-serial default resolution/exposure/gain tables, `configure_all_cameras()`. |
| `video_writers.py` | `VideoWriterManager` — ffmpeg and OpenCV writer backends, pipe tuning. |
| `timestamp_utils.py` | `latch_timestamp_mapping()`, `trim_timestamp_zeros()`, `save_timestamps()`. |
| `grab_loops.py` | `GrabLoopRunner` — frame retrieval loop and the three grab modes. |
| `logging_config.py` | `get_camera_logger()` — file + console logger with graceful fallback. |
| `postprocess.py` | Post-recording sync, file moves, and `combine_videos` call. |
| `synchronization/` | Timestamp synchronization across Basler and Pupil streams. |
| `intrinsics/` | Per-camera lens calibration and distortion correction. |
| `diagnostics/` | Timestamp plotting and file exploration tools. |

---

## Multicamera recording

1. Open `src/cameras/run_recording.py` in VSCode/Cursor.
2. Edit the **`CONFIG`** section near the top:
   - `RECORDING_NAME` — set to match your naming schema (e.g. `ferret_416_P51_E12`). If this is a calibration session use `"calibration"`.
   - `FPS` — acquisition frame rate (default `90`).
   - `BINNING_FACTOR` — `1` for full resolution, `2` for half (default `2`).
   - `HARDWARE_TRIGGER` — `True` if cameras are slaved to an external TTL trigger.
   - `NIR_ONLY` — `True` to record only NIR cameras; `False` for all selected cameras.
3. Uncomment **exactly one** grab mode at the bottom of `main()`:
   - `grab_n_frames(n)` — record exactly n frames on every camera.
   - `grab_n_seconds(t)` — record for t seconds (e.g. `2.5 * 60` for 2.5 minutes).
   - `grab_until_input()` — record until you press Enter in the terminal. Use this when running Basler and Pupil together and stopping both manually.
4. Run the file:
   ```bash
   uv run python src/cameras/run_recording.py
   ```
   or press the **Run** button in VSCode/Cursor.
5. Recordings are saved to `/home/scholl-lab/recordings/<session_date>/<RECORDING_NAME>/raw_videos/`.

> **Recalibrate** if it is the first recording of the day or any camera has been moved.

---

## Pupil eye recording

1. Open a new terminal.
2. Run:
   ```bash
   conda activate pupil_source
   cd Documents/git_repos/pupil/pupil_src
   python main.py capture
   ```
3. The Pupil GUI will open. If eye cameras do not appear automatically, go to **Settings → Detect Eye 0 / Detect Eye 1**.
4. Set the recording name under **`Recorder: Recording session name`** — use the same name as the Basler `RECORDING_NAME`.
5. Press `R` on screen or `r` on the keyboard to start/stop recording.
6. Recordings are saved to `/home/scholl-lab/pupil_recordings`.

---

## Synchronizing Basler and Pupil

After both recordings are complete, run postprocessing to align timestamps and move files into the standard session layout:

```bash
uv run python src/cameras/postprocess.py
```

This calls `TimestampSynchronize`, `TimestampConverter`, and `combine_videos` to produce the `full_recording/` directory structure expected by the rest of the pipeline.

---

## Camera configuration

Per-camera exposure, gain, and resolution defaults live in `camera_config.py`:

```python
# camera_config.py
SERIAL_TO_EXPOSURE_GAIN = {
    "24908831": (5000, 1.0),
    "24908832": (5000, 0.0),
    ...
}
```

To override defaults for a session, pass the `overrides` argument in `run_recording.py`:

```python
configure_all_cameras(
    camera_array=mcr.camera_array,
    devices=mcr.devices,
    overrides={"24908831": (3000, 2.0)},  # custom exposure + gain for one camera
)
```

---

## Resources

- pypylon getting started: https://pythonforthelab.com/blog/getting-started-with-basler-cameras/
- Basler camera parameter reference: https://docs.baslerweb.com/pylonapi/net/T_Basler_Pylon_PLCamera
