"""
MultiCameraRecording — orchestrator for Basler multi-camera acquisition.

This class is the single public entry point for recording code. It owns the
camera array and wires together the focused subsystems:

    camera_config   — CameraProfile, CAMERAS, KNOWN_SERIALS, and derived helpers
    video_writers   — VideoWriterManager (OpenCV or ffmpeg backends)
    timestamp_utils — hardware timestamp latching and saving
    grab_loops      — GrabLoopRunner (frame retrieval loop)

Run a recording session via run_recording.py (preferred) or call this class
directly from your own script.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pypylon.pylon as pylon

from src.cameras.camera_config import (
    ImageShape,
    KNOWN_SERIALS,
    NO_BINNING_SERIALS,
    get_image_shape,
)
from src.cameras.grab_loops import GrabLoopRunner
from src.cameras.video_writers import VideoWriterManager
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session folder helper
# ---------------------------------------------------------------------------

def make_session_folder_at_base_path(base_path: Path) -> Path:
    """
    Create and return a date-stamped session folder under base_path.

    Example: base_path/session_2025-07-11/
    """
    now = datetime.now()
    folder_name = f"session_{now.year}-{now.month:02}-{now.day:02}"
    session_path = base_path / folder_name
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


# ---------------------------------------------------------------------------
# MultiCameraRecording
# ---------------------------------------------------------------------------

class MultiCameraRecording:
    """
    Orchestrates Basler multi-camera acquisition.

    Responsibilities:
    - Enumerate and open cameras.
    - Manage image format converters.
    - Delegate video writing to VideoWriterManager.
    - Delegate grab loops to GrabLoopRunner.
    - Expose camera configuration helpers (FPS, binning, triggering, exposure, gain).
    """

    def __init__(
        self,
        output_path: Path = Path(__file__).parent,
        nir_only: bool = True,
        fps: float = 30.0,
    ) -> None:
        self.tlf = pylon.TlFactory.GetInstance()

        all_devices = list(self.tlf.EnumerateDevices())
        nir_devices = [d for d in all_devices if "NIR" in d.GetModelName()]
        select_devices = [d for d in all_devices if d.GetSerialNumber() in KNOWN_SERIALS]

        self.devices = nir_devices if nir_only else select_devices
        self.camera_array = self._create_camera_array()

        self.fps = fps
        self._setup_image_format_converters()
        self.image_shapes = self._build_image_shapes()
        self._validate_output_path(output_path)

        self._writer: VideoWriterManager | None = None
        self._grabber: GrabLoopRunner | None = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _create_camera_array(self) -> pylon.InstantCameraArray:
        camera_array = pylon.InstantCameraArray(len(self.devices))
        for index, cam in enumerate(camera_array):
            cam.Attach(self.tlf.CreateDevice(self.devices[index]))
        return camera_array

    def _setup_image_format_converters(self) -> None:
        self.rgb_converter = pylon.ImageFormatConverter()
        self.nir_converter = pylon.ImageFormatConverter()
        self.rgb_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.rgb_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.nir_converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.nir_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def _build_image_shapes(self) -> dict[int, ImageShape]:
        shapes: dict[int, ImageShape] = {}
        for index, camera in enumerate(self.camera_array):
            serial = self.devices[index].GetSerialNumber()
            shapes[camera.GetCameraContext()] = get_image_shape(serial)
        return shapes

    def _validate_output_path(self, output_path: Path) -> None:
        output_path = Path(output_path)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path must be a directory, got file: {output_path}")

        while (
            output_path != Path(__file__).parent
            and output_path.exists()
            and next(output_path.iterdir(), None)
        ):
            stem = output_path.stem
            parts = stem.split("__")
            if len(parts) > 1:
                parts[-1] = str(int(parts[-1]) + 1)
            else:
                parts.append("1")
            output_path = output_path.parent / "__".join(parts)

        if output_path.stem != "raw_videos":
            output_path = output_path / "raw_videos"

        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Videos will be saved to {output_path}")
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Camera array lifecycle
    # ------------------------------------------------------------------

    def open_camera_array(self) -> None:
        if not self.camera_array.IsOpen():
            self.camera_array.Open()
            index_to_serial: dict[int, str] = {}
            for index, camera in enumerate(self.camera_array):
                serial = camera.DeviceInfo.GetSerialNumber()
                camera.SetCameraContext(index)
                index_to_serial[index] = serial
                logger.info(f"Camera context {index} → serial {serial}")

            with open(self.output_path / "index_to_serial_number_mapping.json", mode="x") as f:
                json.dump(index_to_serial, f, indent=4)

    def close_camera_array(self) -> None:
        self.camera_array.Close()

    # ------------------------------------------------------------------
    # Camera configuration
    # ------------------------------------------------------------------

    def set_hardware_triggering(self, hardware_triggering: bool = False) -> None:
        for camera in self.camera_array:
            if hardware_triggering:
                camera.TriggerMode.Value = "On"
                camera.TriggerSource.Value = "Line3"
                camera.TriggerActivation.Value = "RisingEdge"
            else:
                camera.TriggerMode.Value = "Off"
            logger.info(f"Camera {camera.GetCameraContext()} trigger: {camera.TriggerMode.Value}")

    def set_max_num_buffer(self, num: int) -> None:
        for cam in self.camera_array:
            cam.MaxNumBuffer.Value = num

    def set_fps(self, fps: float) -> None:
        self.fps = fps
        if self._writer is not None:
            self._writer.update_fps(fps)

    def set_exposure_time(self, camera: pylon.InstantCamera, exposure_time: int) -> None:
        camera.ExposureTime.Value = exposure_time
        logger.info(
            f"Camera {camera.GetCameraContext()} exposure: {exposure_time} {camera.ExposureTime.Unit}"
        )

    def set_gain(self, camera: pylon.InstantCamera, gain: float) -> None:
        camera.Gain.Value = gain
        logger.info(f"Camera {camera.GetCameraContext()} gain: {gain}")

    def set_image_resolution(self, binning_factor: int) -> None:
        if binning_factor not in (1, 2, 3, 4):
            raise ValueError(f"Valid binning factors are 1-4, got {binning_factor}")

        for cam in self.camera_array:
            serial = self.devices[cam.GetCameraContext()].GetSerialNumber()
            if serial in NO_BINNING_SERIALS:
                cam.BinningHorizontal.Value = 1
                cam.BinningVertical.Value = 1
                continue

            cam.BinningHorizontal.Value = binning_factor
            cam.BinningVertical.Value = binning_factor
            ctx = cam.GetCameraContext()
            original = self.image_shapes[ctx]
            updated = ImageShape(
                width=original.width // binning_factor,
                height=original.height // binning_factor,
            )
            self.image_shapes[ctx] = updated
            logger.info(f"Camera {ctx} binning {binning_factor}x → {updated.width}x{updated.height}")

        if self._writer is not None:
            self._writer.update_image_shapes(self.image_shapes)

    def camera_information(self) -> None:
        """Log hardware info for all cameras."""
        for cam in self.camera_array:
            logger.info(
                f"Camera {cam.GetCameraContext()} — "
                f"max buffers: {cam.MaxNumBuffer.Value}, "
                f"buffer size: {cam.StreamGrabber.MaxBufferSize.Value}, "
                f"exposure: {cam.ExposureTime.Value}, "
                f"fps: {cam.AcquisitionFrameRate.Value}, "
                f"gain: {cam.Gain.Value}"
            )

    # ------------------------------------------------------------------
    # Video writer convenience methods (delegate to VideoWriterManager)
    # ------------------------------------------------------------------

    def create_video_writers_ffmpeg(self) -> None:
        self._writer = VideoWriterManager(
            camera_array=self.camera_array,
            image_shapes=self.image_shapes,
            fps=self.fps,
            output_path=self.output_path,
        )
        self._writer.create_ffmpeg()

    def create_video_writers(self) -> None:
        self._writer = VideoWriterManager(
            camera_array=self.camera_array,
            image_shapes=self.image_shapes,
            fps=self.fps,
            output_path=self.output_path,
        )
        self._writer.create_opencv()

    # ------------------------------------------------------------------
    # Grab modes (delegate to GrabLoopRunner)
    # ------------------------------------------------------------------

    def _get_grabber(self) -> GrabLoopRunner:
        if self._writer is None:
            raise RuntimeError("Call create_video_writers_ffmpeg() before grabbing.")
        return GrabLoopRunner(
            camera_array=self.camera_array,
            n_cameras=len(self.devices),
            fps=self.fps,
            writer=self._writer,
            output_path=self.output_path,
        )

    def grab_n_frames(self, number_of_frames: int) -> None:
        self._get_grabber().grab_n_frames(number_of_frames)

    def grab_n_seconds(self, number_of_seconds: float) -> None:
        self._get_grabber().grab_n_seconds(number_of_seconds)

    def grab_until_input(self) -> None:
        self._get_grabber().grab_until_input()

    def pylon_internal_statistics(self) -> bool:
        return self._get_grabber().pylon_internal_statistics()
