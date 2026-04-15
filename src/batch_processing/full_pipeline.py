"""
process the entire pipeline in one go
use boolean parameters to turn steps on and off

requires the following repos/bracnhes installed:
    skellyclicker: https://github.com/freemocap/skellyclicker
    dlc_to_3d: https://github.com/philipqueen/freemocap_playground@philip/bs
    freemocap: https://github.com/freemocap/freemocap
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from src.batch_processing.postprocess_recording import process_recording
from src.cameras.postprocess import postprocess
from src.ferret_gaze.realtime import (
    build_synthetic_replay_packets,
    compare_stub_solvers,
    create_inference_runtime,
    create_realtime_publisher,
    create_triangulator,
    discover_session_calibration_toml,
    format_latency_summary,
    load_realtime_runtime_config,
    run_realtime_compute_scaffold,
    run_realtime_transport_scaffold,
)
from src.ferret_gaze.realtime.anatomical_mocap_fuse import create_eye_calibrator, create_gaze_fuser
from src.ferret_gaze.realtime.kabsch_skull_solver import create_skull_solver
from src.ferret_gaze.realtime.live_mocap_grab_session import run_live_mocap_grab_n_frames_publish
from src.ferret_gaze.realtime.live_mocap_pipeline import (
    build_synthetic_live_mocap_frame_sets,
    run_live_mocap_compute_publish_session,
)
from src.ferret_gaze.realtime.runtime_config import RealtimeRuntimeConfig
from src.utilities.folder_utilities.recording_folder import RecordingFolder
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


HEAD_DLC_ITERATION = 17
EYE_DLC_ITERATION = 30
TOY_DLC_ITERATION = 10


def _dlc_metadata_is_outdated(dlc_output_folder: Path | None, required_iteration: int) -> bool:
    """Return True if skellyclicker_metadata.json exists and has a lower iteration than required."""
    if dlc_output_folder is None:
        return True
    metadata_path = dlc_output_folder / "skellyclicker_metadata.json"
    if not metadata_path.exists():
        return True
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata.get("iteration", 0) < required_iteration


def _make_clean_env() -> dict:
    """Return a copy of the environment with Python venv vars stripped out."""
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    return env


def _spawn_pty_process(command_list: list, clean_env: dict) -> tuple:
    """
    Open a PTY, size it to match the parent terminal, and spawn the command.

    Returns (process, master_fd) so the caller can drain output and close the fd.
    Falls back silently if stdout is not a TTY when setting the window size.
    """
    import pty, fcntl, termios, struct
    master_fd, slave_fd = pty.openpty()
    # Match PTY window size to parent terminal so tqdm sizes its bar correctly.
    try:
        term_size = os.get_terminal_size(sys.stdout.fileno())
        winsize = struct.pack("HHHH", term_size.lines, term_size.columns, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
    except OSError:
        pass
    process = subprocess.Popen(
        command_list, env=clean_env,
        stdout=slave_fd, stderr=slave_fd, stdin=subprocess.DEVNULL,
    )
    os.close(slave_fd)
    return process, master_fd


def _drain_pty_fd(master_fd: int) -> None:
    """Stream all output from the PTY master fd to stdout until the child closes it."""
    try:
        while True:
            try:
                data = os.read(master_fd, 4096)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
            except OSError:
                # Raised when the slave end is closed (process exited)
                break
    finally:
        os.close(master_fd)


def _run_subprocess_streaming(command_list: list, clean_env: dict, use_pty: bool = False) -> None:
    """
    Run a subprocess and stream its output in real time, raising on non-zero exit.

    Args:
        command_list: Command and arguments to run.
        clean_env: Environment dict for the subprocess.
        use_pty: If True, allocate a PTY so tqdm stays in single-line overwrite
                 mode instead of printing every update as a new line.
    """
    if use_pty and sys.platform != "win32":
        process, master_fd = _spawn_pty_process(command_list, clean_env)
        _drain_pty_fd(master_fd)
    else:
        process = subprocess.Popen(
            command_list, env=clean_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command_list[0])


def run_skellyclicker_subprocess(
        recording_folder_path: Path,
        venv_path: str = "/home/scholl-lab/anaconda3/envs/skellyclicker/bin/python",
        script_path: str = "/home/scholl-lab/skellyclicker/skellyclicker/scripts/process_recording.py",
        include_eye: bool = True,
    ):
    command_list = [venv_path, "-u", script_path, recording_folder_path]
    if not include_eye:
        command_list.append("--skip-eye")
    _run_subprocess_streaming(command_list, _make_clean_env(), use_pty=True)


def run_triangulation_subprocess(
        recording_folder_path: Path,
        calibration_toml_path: Path,
        venv_path: str = "/home/scholl-lab/Documents/git_repos/dlc_to_3d/.venv/bin/python",
        script_path: str = "/home/scholl-lab/Documents/git_repos/dlc_to_3d/dlc_reconstruction/dlc_to_3d.py",
        skip_toy: bool = False,
    ):
    command_list = [venv_path, script_path, recording_folder_path, calibration_toml_path]
    if skip_toy:
        command_list.append("--skip-toy")
    _run_subprocess_streaming(command_list, _make_clean_env())


def run_calibration_subprocess(
        calibration_videos_path: Path,
        venv_path: str = "/home/scholl-lab/anaconda3/envs/fmc/bin/python",
        script_path: str = "/home/scholl-lab/Documents/git_repos/freemocap/experimental/batch_process/headless_calibration.py",
    ):
    command_list = [
        venv_path, script_path, calibration_videos_path,
        "--square-size", "57", "--5x3", "--use-groundplane",
    ]
    _run_subprocess_streaming(command_list, _make_clean_env())



def _resolve_overwrite_flags(
    recording_folder: "RecordingFolder",
    overwrite_synchronization: bool,
    overwrite_calibration: bool,
    overwrite_dlc: bool,
    overwrite_triangulation: bool,
    overwrite_eye_postprocessing: bool,
    overwrite_skull_postprocessing: bool,
    overwrite_gaze: bool,
) -> dict:
    """
    Propagate overwrite flags through dependent pipeline steps and force DLC
    reprocessing if any outputs were produced with an outdated model iteration.
    Returns a dict of resolved flag values keyed by step name.
    """
    if not overwrite_dlc and (
        _dlc_metadata_is_outdated(recording_folder.head_body_dlc_output, HEAD_DLC_ITERATION)
        or _dlc_metadata_is_outdated(recording_folder.eye_dlc_output, EYE_DLC_ITERATION)
        or _dlc_metadata_is_outdated(recording_folder.toy_dlc_output, TOY_DLC_ITERATION)
    ):
        logger.warning("DLC outputs are from an outdated model iteration, forcing DLC reprocessing")
        overwrite_dlc = True

    if overwrite_synchronization:
        overwrite_calibration = True
    if overwrite_dlc:
        overwrite_eye_postprocessing = True
        if overwrite_calibration:
            overwrite_triangulation = True
    if overwrite_triangulation:
        overwrite_skull_postprocessing = True
    if overwrite_eye_postprocessing or overwrite_skull_postprocessing:
        overwrite_gaze = True

    return dict(
        synchronization=overwrite_synchronization,
        calibration=overwrite_calibration,
        dlc=overwrite_dlc,
        triangulation=overwrite_triangulation,
        eye_postprocessing=overwrite_eye_postprocessing,
        skull_postprocessing=overwrite_skull_postprocessing,
        gaze=overwrite_gaze,
    )


def _run_postprocessing(
    recording_folder: "RecordingFolder",
    recording_folder_path: Path,
    include_eye: bool,
    flags: dict,
) -> None:
    """Run eye/skull/gaze postprocessing steps that are flagged or not yet done."""
    run_eye = include_eye and (flags["eye_postprocessing"] or not recording_folder.is_eye_postprocessed())
    run_skull = flags["skull_postprocessing"] or not recording_folder.is_skull_postprocessed()
    run_gaze = include_eye and (flags["gaze"] or not recording_folder.is_gaze_postprocessed())

    if run_eye or run_skull or run_gaze:
        logger.info("Running gaze processing...")
        process_recording(
            recording_folder=recording_folder,
            skip_eye=not run_eye,
            skip_skull=not run_skull,
            skip_gaze=not run_gaze,
        )
    recording_folder.check_eye_postprocessing()
    recording_folder.check_skull_postprocessing()
    recording_folder.check_gaze_postprocessing()
    logger.info("Gaze calculations complete")
    logger.info("Session processed: %s", recording_folder_path)


def _run_offline_pipeline(
    recording_folder_path: Path,
    calibration_toml_path: Path | None = None,
    include_eye: bool = True,
    overwrite_synchronization: bool = False,
    overwrite_calibration: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False,
):
    """Run the existing batch/offline pipeline end-to-end."""
    recording_folder = RecordingFolder.from_folder_path(folder=recording_folder_path)
    flags = _resolve_overwrite_flags(
        recording_folder,
        overwrite_synchronization, overwrite_calibration, overwrite_dlc,
        overwrite_triangulation, overwrite_eye_postprocessing,
        overwrite_skull_postprocessing, overwrite_gaze,
    )

    # Synchronization
    if flags["synchronization"] or not recording_folder.is_synchronized():
        logger.info("Synchronizing videos at %s", recording_folder.base_recordings_folder)
        postprocess(session_folder_path=recording_folder.base_recordings_folder, include_eyes=include_eye)
    recording_folder.check_synchronization()
    logger.info("Synchronizing videos completed")

    # Calibration
    if flags["calibration"] or not recording_folder.is_calibrated():
        logger.info("Calibrating session...")
        run_calibration_subprocess(calibration_videos_path=recording_folder.calibration_videos)
    recording_folder.check_calibration()
    logger.info("Calibration complete")

    # DLC
    if flags["dlc"] or not recording_folder.is_dlc_processed():
        logger.info("Running pose estimation...")
        run_skellyclicker_subprocess(recording_folder_path=recording_folder_path)
    recording_folder.check_dlc_output()
    logger.info("Pose estimation complete")

    # Triangulation
    if flags["triangulation"] or not recording_folder.is_triangulated():
        if calibration_toml_path is None:
            calibration_toml_path = recording_folder.calibration_toml_path
        if calibration_toml_path is None:
            raise ValueError("No calibration toml file found, could not run triangulation")
        logger.info("Running triangulation...")
        run_triangulation_subprocess(
            recording_folder_path=recording_folder_path,
            calibration_toml_path=calibration_toml_path,
        )
    recording_folder.check_triangulation()
    logger.info("Triangulation complete")

    _run_postprocessing(recording_folder, recording_folder_path, include_eye, flags)


def _resolve_realtime_calibration_toml(
    recording_folder_path: Path,
    calibration_toml_path: Path | None,
    runtime_config: RealtimeRuntimeConfig,
) -> Path | None:
    """Prefer explicit config path, then caller arg, then session calibration discovery."""
    if runtime_config.calibration_toml_path:
        return Path(runtime_config.calibration_toml_path)
    if calibration_toml_path is not None:
        return calibration_toml_path
    return discover_session_calibration_toml(recording_folder_path)


def _run_realtime_pipeline(
    recording_folder_path: Path,
    calibration_toml_path: Path | None = None,
    realtime_config_path: Path | None = None,
    include_eye: bool = True,
    overwrite_synchronization: bool = False,
    overwrite_calibration: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False,
) -> None:
    """
    Realtime mode: ``scaffold`` (synthetic transport + replay compute) or
    ``live_mocap`` (frame bundles -> infer -> triangulate -> publish).
    """
    # Keep signature parity with offline mode so a single top-level API can
    # switch behavior without changing caller argument shapes.
    _ = (
        include_eye,
        overwrite_synchronization,
        overwrite_calibration,
        overwrite_dlc,
        overwrite_triangulation,
        overwrite_eye_postprocessing,
        overwrite_skull_postprocessing,
        overwrite_gaze,
    )
    runtime_config = load_realtime_runtime_config(config_path=realtime_config_path)

    inference_runtime = create_inference_runtime(
        backend=runtime_config.inference_backend,
        model_path=Path(runtime_config.inference_model_path)
        if runtime_config.inference_model_path
        else None,
        provider=runtime_config.onnx_provider,
        images_input_height=runtime_config.inference_images_height,
        images_input_width=runtime_config.inference_images_width,
        output_uv_normalized=runtime_config.inference_output_uv_normalized,
    )
    calib_resolved = _resolve_realtime_calibration_toml(
        recording_folder_path,
        calibration_toml_path,
        runtime_config,
    )
    triangulator = create_triangulator(
        backend=runtime_config.triangulation_backend,
        calibration_toml_path=calib_resolved,
    )
    skull_solver = create_skull_solver(
        runtime_config.skull_solver_backend,
        kabsch_reference_npy=Path(runtime_config.skull_solver_kabsch_reference_npy)
        if runtime_config.skull_solver_kabsch_reference_npy
        else None,
    )
    eye_calibrator = create_eye_calibrator(
        runtime_config.eye_calibrator_backend,
        vergence_ema_alpha=runtime_config.anatomical_vergence_ema_alpha,
    )
    gaze_fuser = create_gaze_fuser(
        runtime_config.gaze_fuser_backend,
        half_ipd_mm=runtime_config.anatomical_half_ipd_mm,
        eye_y_mm=runtime_config.anatomical_eye_y_mm,
        eye_z_mm=runtime_config.anatomical_eye_z_mm,
    )

    # Step 4 benchmark gate scaffold: compare stub solvers on a shared replay stream.
    replay_packets = build_synthetic_replay_packets(n_packets=runtime_config.benchmark_packets)
    reference_packets = [packet.model_copy(deep=True) for packet in replay_packets]
    comparison = compare_stub_solvers(
        replay_packets=replay_packets,
        reference_packets=reference_packets,
    )
    logger.info(
        "Stub solver benchmark: ukf(lat_p95_ms=%.2f,pos_err_mm=%.4f) "
        "vs ceres(lat_p95_ms=%.2f,pos_err_mm=%.4f) -> recommended=%s",
        comparison.ukf.p95_solver_latency_ms,
        comparison.ukf.mean_position_error_mm,
        comparison.ceres.p95_solver_latency_ms,
        comparison.ceres.mean_position_error_mm,
        comparison.recommended_solver,
    )

    if runtime_config.realtime_mode == "live_mocap":
        publisher = create_realtime_publisher(
            backend=runtime_config.transport_backend,
            endpoint=runtime_config.transport_endpoint,
            topic=runtime_config.transport_topic,
            payload_format=runtime_config.transport_payload_format,
        )
        session_closed_publisher = False
        try:
            if runtime_config.live_mocap_frame_source == "synthetic":
                logger.info(
                    "Starting live_mocap session (synthetic frame bundles, ticks=%d)",
                    runtime_config.transport_packets,
                )
                frame_sets = build_synthetic_live_mocap_frame_sets(
                    runtime_config.transport_packets,
                    n_cams=runtime_config.live_mocap_synthetic_camera_count,
                    height=runtime_config.live_mocap_synthetic_height,
                    width=runtime_config.live_mocap_synthetic_width,
                )
                run_live_mocap_compute_publish_session(
                    frame_sets=frame_sets,
                    publisher=publisher,
                    inference_runtime=inference_runtime,
                    triangulator=triangulator,
                    hz=runtime_config.transport_hz,
                    stale_threshold_ms=runtime_config.stale_threshold_ms,
                    calibrator=eye_calibrator,
                    fuser=gaze_fuser,
                    skull_solver=skull_solver,
                )
                session_closed_publisher = True
            elif runtime_config.live_mocap_frame_source == "grab":
                grab_out = (
                    Path(runtime_config.live_mocap_grab_output_path)
                    if runtime_config.live_mocap_grab_output_path
                    else recording_folder_path
                )
                n_frames = runtime_config.live_mocap_grab_n_frames or runtime_config.transport_packets
                grab_fps = (
                    runtime_config.live_mocap_grab_fps
                    if runtime_config.live_mocap_grab_fps is not None
                    else runtime_config.transport_hz
                )
                logger.info(
                    "Starting live_mocap grab session: output=%s, n_frames=%d, fps=%.2f",
                    grab_out,
                    n_frames,
                    grab_fps,
                )
                run_live_mocap_grab_n_frames_publish(
                    output_path=grab_out,
                    nir_only=runtime_config.live_mocap_grab_nir_only,
                    fps=float(grab_fps),
                    binning_factor=runtime_config.live_mocap_grab_binning_factor,
                    hardware_triggering=runtime_config.live_mocap_grab_hardware_trigger,
                    n_frames=int(n_frames),
                    publisher=publisher,
                    inference_runtime=inference_runtime,
                    triangulator=triangulator,
                    stale_threshold_ms=runtime_config.stale_threshold_ms,
                    wire_queue_size=runtime_config.live_mocap_grab_wire_queue_size,
                    pace_hz=runtime_config.live_mocap_grab_pace_hz,
                    calibrator=eye_calibrator,
                    fuser=gaze_fuser,
                    skull_solver=skull_solver,
                )
            else:
                raise ValueError(
                    f"Unsupported live_mocap_frame_source: {runtime_config.live_mocap_frame_source!r}"
                )
        finally:
            if not session_closed_publisher:
                publisher.close()
        logger.info("Realtime live_mocap session complete")
        return

    if runtime_config.realtime_mode != "scaffold":
        raise ValueError(f"Unsupported realtime_mode: {runtime_config.realtime_mode!r}")

    logger.info("Starting realtime transport scaffold (synthetic packets)")
    publisher = create_realtime_publisher(
        backend=runtime_config.transport_backend,
        endpoint=runtime_config.transport_endpoint,
        topic=runtime_config.transport_topic,
        payload_format=runtime_config.transport_payload_format,
    )
    summary = run_realtime_transport_scaffold(
        publisher=publisher,
        n_packets=runtime_config.transport_packets,
        hz=runtime_config.transport_hz,
        stale_threshold_ms=runtime_config.stale_threshold_ms,
    )
    logger.info(format_latency_summary(summary))

    # Step 6 scaffold: run per-frame compute stages on replay packets.
    compute_input = replay_packets[: runtime_config.compute_packets]
    computed_packets = run_realtime_compute_scaffold(
        compute_input,
        inference_runtime=inference_runtime,
        triangulator=triangulator,
    )
    mean_confidence = (
        sum(packet.confidence or 0.0 for packet in computed_packets) / len(computed_packets)
        if computed_packets
        else 0.0
    )
    logger.info(
        "Per-frame compute scaffold complete: packets=%d, mean_confidence=%.4f",
        len(computed_packets),
        mean_confidence,
    )
    logger.info("Realtime transport scaffold complete")


def run_pipeline(
    recording_folder_path: Path,
    calibration_toml_path: Path | None = None,
    realtime_config_path: Path | None = None,
    include_eye: bool = True,
    overwrite_synchronization: bool = False,
    overwrite_calibration: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False,
    mode: Literal["offline", "realtime"] = "offline",
) -> None:
    """
    Top-level pipeline entrypoint with mode switching.

    `offline` preserves the existing batch behavior.
    `realtime` is scaffolded and reserved for the live Unreal path.
    Pass `realtime_config_path` to override realtime runtime defaults from JSON.
    """
    if mode == "offline":
        _run_offline_pipeline(
            recording_folder_path=recording_folder_path,
            calibration_toml_path=calibration_toml_path,
            include_eye=include_eye,
            overwrite_synchronization=overwrite_synchronization,
            overwrite_calibration=overwrite_calibration,
            overwrite_dlc=overwrite_dlc,
            overwrite_triangulation=overwrite_triangulation,
            overwrite_eye_postprocessing=overwrite_eye_postprocessing,
            overwrite_skull_postprocessing=overwrite_skull_postprocessing,
            overwrite_gaze=overwrite_gaze,
        )
        return
    if mode == "realtime":
        _run_realtime_pipeline(
            recording_folder_path=recording_folder_path,
            calibration_toml_path=calibration_toml_path,
            realtime_config_path=realtime_config_path,
            include_eye=include_eye,
            overwrite_synchronization=overwrite_synchronization,
            overwrite_calibration=overwrite_calibration,
            overwrite_dlc=overwrite_dlc,
            overwrite_triangulation=overwrite_triangulation,
            overwrite_eye_postprocessing=overwrite_eye_postprocessing,
            overwrite_skull_postprocessing=overwrite_skull_postprocessing,
            overwrite_gaze=overwrite_gaze,
        )
        return
    raise ValueError("mode must be either 'offline' or 'realtime'")


def full_pipeline(
    recording_folder_path: Path,
    calibration_toml_path: Path | None = None,
    include_eye: bool = True,
    overwrite_synchronization: bool = False,
    overwrite_calibration: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False,
) -> None:
    """
    Backward-compatible alias for the existing offline batch pipeline.

    New callers should use `run_pipeline(..., mode='offline'|'realtime')`.
    """
    run_pipeline(
        recording_folder_path=recording_folder_path,
        calibration_toml_path=calibration_toml_path,
        include_eye=include_eye,
        overwrite_synchronization=overwrite_synchronization,
        overwrite_calibration=overwrite_calibration,
        overwrite_dlc=overwrite_dlc,
        overwrite_triangulation=overwrite_triangulation,
        overwrite_eye_postprocessing=overwrite_eye_postprocessing,
        overwrite_skull_postprocessing=overwrite_skull_postprocessing,
        overwrite_gaze=overwrite_gaze,
        mode="offline",
    )


if __name__=="__main__":
    recording_folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-22_ferret_420_EO13/full_recording"
    )

    if "clips" not in str(recording_folder_path) and "full_recording" not in str(recording_folder_path):
        recording_folder_path = recording_folder_path / "full_recording"


    # Allow one-shot setup even when parent directories are missing.
    recording_folder_path.mkdir(exist_ok=True, parents=True)
    (recording_folder_path / "mocap_data").mkdir(exist_ok=True, parents=True)
    (recording_folder_path / "eye_data").mkdir(exist_ok=True, parents=True)
    logger.info("Processing %s", recording_folder_path)

    run_pipeline(
        recording_folder_path=recording_folder_path,
        overwrite_synchronization=False,
        overwrite_calibration=False,
        overwrite_dlc=False,
        overwrite_triangulation=False,
        overwrite_eye_postprocessing=True,
        overwrite_skull_postprocessing=False,
        overwrite_gaze=True,
        mode="offline",
    )
