"""
process the entire pipeline in one go
use boolean parameters to turn steps on and off

requires the following repos/bracnhes installed:
    skellyclicker: https://github.com/freemocap/skellyclicker
    dlc_to_3d: https://github.com/philipqueen/freemocap_playground@philip/bs
    freemocap: https://github.com/freemocap/freemocap
"""
from pathlib import Path
import json
import subprocess
import os
import sys

from python_code.batch_processing.postprocess_recording import process_recording
from python_code.cameras.postprocess import postprocess
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.utilities.logging_config import get_logger

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
):
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


if __name__=="__main__":
    recording_folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-22_ferret_420_EO13/full_recording"
    )

    if "clips" not in str(recording_folder_path) and "full_recording" not in str(recording_folder_path):
        recording_folder_path = recording_folder_path / "full_recording"


    recording_folder_path.mkdir(exist_ok=True, parents=False)
    (recording_folder_path / "mocap_data").mkdir(exist_ok=True, parents=False)
    (recording_folder_path / "eye_data").mkdir(exist_ok=True, parents=False)
    logger.info("Processing %s", recording_folder_path)

    full_pipeline(
        recording_folder_path=recording_folder_path,
        overwrite_synchronization=False,
        overwrite_calibration=False,
        overwrite_dlc=False,
        overwrite_triangulation=False,
        overwrite_eye_postprocessing=True,
        overwrite_skull_postprocessing=False,
        overwrite_gaze=True
    )
