"""
Lab recording entry point.

Edit the variables in the CONFIG section below, then run:

    uv run python python_code/cameras/run_recording.py

or press the Run button in VSCode/Cursor.

CONFIG variables:
    BASE_PATH         Root directory where session folders are created.
    RECORDING_NAME    Sub-folder name for this recording (e.g. ferret ID + session tag).
    FPS               Acquisition frame rate in Hz.
    BINNING_FACTOR    Spatial binning: 1 (full res), 2, 3, or 4.
    HARDWARE_TRIGGER  True to sync cameras to external TTL trigger; False for free-run.
    NIR_ONLY          True to record only NIR cameras; False to include all select cameras.

GRAB MODE:
    Uncomment exactly one grab call at the bottom of this file.
"""

import os
from pathlib import Path

import psutil

from python_code.cameras.camera_config import configure_all_cameras
from python_code.cameras.multicamera_recording import MultiCameraRecording, make_session_folder_at_base_path

# =============================================================================
# CONFIG — edit these before each recording session
# =============================================================================

BASE_PATH = Path("/home/scholl-lab/recordings")

# RECORDING_NAME = "calibration"
# RECORDING_NAME = "ferret_416_P51_E12"
RECORDING_NAME = "psychopy_trial_1_ferret411_03-19-26"

FPS = 90
BINNING_FACTOR = 2
HARDWARE_TRIGGER = True
NIR_ONLY = False


# =============================================================================
# Helpers
# =============================================================================

def set_high_priority() -> None:
    """Raise this process's scheduling priority to reduce frame-drop risk."""
    process = psutil.Process(os.getpid())
    process.nice(-10)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    set_high_priority()

    output_path = make_session_folder_at_base_path(BASE_PATH) / RECORDING_NAME

    mcr = MultiCameraRecording(output_path=output_path, nir_only=NIR_ONLY, fps=FPS)
    mcr.open_camera_array()

    mcr.set_max_num_buffer(240)
    mcr.set_fps(FPS)
    mcr.set_image_resolution(binning_factor=BINNING_FACTOR)
    mcr.set_hardware_triggering(hardware_triggering=HARDWARE_TRIGGER)

    configure_all_cameras(
        camera_array=mcr.camera_array,
        devices=mcr.devices,
        # Pass overrides here if you need non-default exposure/gain:
        # overrides={"24908831": (3000, 2.0)}
    )

    mcr.camera_information()
    mcr.create_video_writers_ffmpeg()

    # --- Uncomment exactly ONE grab mode ---
    # mcr.grab_n_frames(90)            # e.g. 1 second at 90 fps
    # mcr.grab_n_seconds(2.5 * 60)     # 2.5 minutes
    mcr.grab_until_input()             # press Enter to stop

    mcr.close_camera_array()


if __name__ == "__main__":
    main()
