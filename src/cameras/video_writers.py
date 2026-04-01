"""
Video writer management for multi-camera recording.

Provides VideoWriterManager which abstracts over two backend options:

- OpenCV (cv2.VideoWriter): simpler, lower throughput.
- ffmpeg (via python-ffmpeg): higher throughput, pipe-based, used in production.

Usage:
    manager = VideoWriterManager(camera_array, image_shapes, fps, output_path)
    manager.create_ffmpeg()

    manager.write_frame_ffmpeg(frame, cam_id=0)

    manager.release_ffmpeg()
"""

import fcntl
from src.utilities.logging_config import get_logger
from pathlib import Path
from typing import Union

import cv2
import ffmpeg
import numpy as np
import pypylon.pylon as pylon

from src.cameras.camera_config import ImageShape

logger = get_logger(__name__)

FFMPEG_PIPE_SIZE = 200_000_000  # 200 MB pipe buffer


class VideoWriterManager:
    """
    Manages video writers for all cameras in a recording session.

    Supports both OpenCV and ffmpeg backends. Only one backend should be
    active at a time — call release before switching.
    """

    def __init__(
        self,
        camera_array: pylon.InstantCameraArray,
        image_shapes: dict[int, ImageShape],
        fps: float,
        output_path: Path,
    ) -> None:
        self._camera_array = camera_array
        self._image_shapes = image_shapes
        self._fps = fps
        self._output_path = output_path
        self._opencv_writers: dict[int, cv2.VideoWriter] | None = None
        self._ffmpeg_writers: dict[int, object] | None = None  # subprocess.Popen via ffmpeg

    # ------------------------------------------------------------------
    # OpenCV backend
    # ------------------------------------------------------------------

    def create_opencv(self) -> None:
        """Create one OpenCV VideoWriter per camera."""
        self._opencv_writers = {}
        for index, camera in enumerate(self._camera_array):
            serial = camera.DeviceInfo.GetSerialNumber()
            file_name = f"{serial}.mp4"
            shape = self._image_shapes[camera.GetCameraContext()]

            writer = cv2.VideoWriter(
                str(self._output_path / file_name),
                cv2.VideoWriter.fourcc(*"x265"),
                self._fps,
                (shape.width, shape.height),
                isColor=False,
            )
            self._opencv_writers[index] = writer
        logger.info("OpenCV video writers created.")

    def write_frame_opencv(self, frame: np.ndarray, cam_id: int, frame_number: int) -> None:
        if self._opencv_writers is None:
            raise RuntimeError("OpenCV writers not initialised — call create_opencv() first.")
        writer = self._opencv_writers[cam_id]
        if not writer.isOpened():
            raise RuntimeWarning(f"OpenCV writer for cam {cam_id} is not open.")
        writer.write(frame)
        if not writer.isOpened():
            raise RuntimeWarning(f"Failed to write frame #{frame_number} for cam {cam_id}.")

    def release_opencv(self) -> None:
        if self._opencv_writers:
            for writer in self._opencv_writers.values():
                writer.release()
        self._opencv_writers = None
        logger.info("OpenCV video writers released.")

    # ------------------------------------------------------------------
    # ffmpeg backend
    # ------------------------------------------------------------------

    def create_ffmpeg(self) -> None:
        """Create one ffmpeg pipe-based writer per camera."""
        self._ffmpeg_writers = {}
        for index, camera in enumerate(self._camera_array):
            serial = camera.DeviceInfo.GetSerialNumber()
            file_name = f"{serial}.mp4"
            shape = self._image_shapes[camera.GetCameraContext()]

            logger.info(f"Camera {index} ({serial}): {shape.width}x{shape.height}")

            process = (
                ffmpeg
                .input(
                    "pipe:",
                    framerate=str(self._fps),
                    format="rawvideo",
                    pix_fmt="gray",
                    s=f"{shape.width}x{shape.height}",
                    hwaccel="auto",
                )
                .output(
                    str(self._output_path / file_name),
                    vcodec="libx264",
                    pix_fmt="yuv420p",
                    preset="ultrafast",
                    tune="zerolatency",
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=False)
            )

            fd = process.stdin.fileno()
            logger.debug(f"Cam {index} original pipe size: {fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)}")
            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, FFMPEG_PIPE_SIZE)
            logger.debug(f"Cam {index} updated pipe size: {fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)}")

            self._ffmpeg_writers[index] = process

        logger.info("ffmpeg video writers created.")

    def write_frame_ffmpeg(self, frame: np.ndarray, cam_id: int) -> None:
        if self._ffmpeg_writers is None:
            raise RuntimeError("ffmpeg writers not initialised — call create_ffmpeg() first.")
        self._ffmpeg_writers[cam_id].stdin.write(frame.tobytes())

    def release_ffmpeg(self) -> None:
        if self._ffmpeg_writers:
            for process in self._ffmpeg_writers.values():
                process.stdin.close()
                process.wait()
        self._ffmpeg_writers = None
        logger.info("ffmpeg video writers released.")

    # ------------------------------------------------------------------
    # Convenience: update when FPS or image shapes change mid-session
    # ------------------------------------------------------------------

    def update_fps(self, new_fps: float, backend: str = "ffmpeg") -> None:
        """Recreate writers with a new framerate."""
        self._fps = new_fps
        if backend == "ffmpeg" and self._ffmpeg_writers is not None:
            self.release_ffmpeg()
            self.create_ffmpeg()
        elif backend == "opencv" and self._opencv_writers is not None:
            self.release_opencv()
            self.create_opencv()

    def update_image_shapes(self, new_shapes: dict[int, ImageShape], backend: str = "ffmpeg") -> None:
        """Recreate writers after an image shape change (e.g. binning)."""
        self._image_shapes = new_shapes
        if backend == "ffmpeg" and self._ffmpeg_writers is not None:
            self.release_ffmpeg()
            self.create_ffmpeg()
        elif backend == "opencv" and self._opencv_writers is not None:
            self.release_opencv()
            self.create_opencv()
