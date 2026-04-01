"""
Frame acquisition (grab) loops for multi-camera recording.

Provides GrabLoopRunner which encapsulates the Basler frame-retrieval loop and
the three public grab modes:

- grab_n_frames(n): stop after exactly n frames on every camera.
- grab_n_seconds(t): stop after t seconds (converted to frames at the set FPS).
- grab_until_input(): run until the operator presses Enter in the terminal.

Also provides pylon_internal_statistics() for post-recording diagnostics, and
the camera-level helpers used inside the grab loop (_set_fps_during_grabbing,
disable_throughput_limit).
"""

import logging
import threading
import time
from typing import Callable

import numpy as np
import pypylon.pylon as pylon

from python_code.cameras.timestamp_utils import latch_timestamp_mapping, save_timestamps
from python_code.cameras.video_writers import VideoWriterManager

logger = logging.getLogger(__name__)

# Pre-allocate this many frames when grab_until_input is used (~1 h at 90 fps).
_OPEN_ENDED_MAX_FRAMES = 324_000


class GrabLoopRunner:
    """
    Runs the Basler frame-retrieval loop and handles all acquisition bookkeeping.

    Owns the interaction between the camera array, video writers, and timestamp
    capture. Does not own the camera array or writers — those are passed in.

    Args:
        camera_array: An open and configured InstantCameraArray.
        n_cameras: Number of active cameras.
        fps: Recording frame rate (used to convert seconds → frames).
        writer: Initialised VideoWriterManager (ffmpeg writers must be created).
        output_path: Directory for timestamps output.
    """

    def __init__(
        self,
        camera_array: pylon.InstantCameraArray,
        n_cameras: int,
        fps: float,
        writer: VideoWriterManager,
        output_path,
    ) -> None:
        self._camera_array = camera_array
        self._n_cameras = n_cameras
        self._fps = fps
        self._writer = writer
        self._output_path = output_path

    # ------------------------------------------------------------------
    # Public grab modes
    # ------------------------------------------------------------------

    def grab_n_frames(self, number_of_frames: int) -> None:
        """Record until every camera has captured at least number_of_frames frames."""
        self._run(
            condition=lambda counts: min(counts) >= number_of_frames,
            max_frames=number_of_frames,
        )

    def grab_n_seconds(self, number_of_seconds: float) -> None:
        """Record for number_of_seconds, inferred from the configured FPS."""
        n = int(number_of_seconds * self._fps)
        self._run(
            condition=lambda counts: min(counts) >= n,
            max_frames=n,
        )

    def grab_until_input(self) -> None:
        """Record until the operator presses Enter in the terminal."""
        stop_event = threading.Event()

        def _wait_for_enter(event: threading.Event) -> None:
            input("Press Enter to stop recording...\n")
            event.set()

        input_thread = threading.Thread(target=_wait_for_enter, args=(stop_event,))
        input_thread.start()

        self._run(
            condition=lambda counts: stop_event.is_set(),
            max_frames=_OPEN_ENDED_MAX_FRAMES,
        )
        input_thread.join()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pylon_internal_statistics(self) -> bool:
        """
        Log Basler internal buffer statistics and return True if no frames dropped.

        Should be called after StopGrabbing.
        """
        successful = True
        for cam in self._camera_array:
            total = cam.StreamGrabber.Statistic_Total_Buffer_Count.GetValue()
            failed = cam.StreamGrabber.Statistic_Failed_Buffer_Count.GetValue()
            logger.info(f"Cam {cam.GetCameraContext()} — total buffers: {total}, failed: {failed}")
            if failed > 0:
                successful = False

        if not successful:
            logger.error(
                "FRAMES WERE DROPPED. "
                "Consider lowering FPS, reducing frame size, or increasing MaxNumBuffer."
            )
        else:
            logger.info("No frames dropped — recording successful.")

        return successful

    # ------------------------------------------------------------------
    # Camera-level setup helpers
    # ------------------------------------------------------------------

    def set_fps_on_cameras(self) -> None:
        """Enable and apply the acquisition frame rate on all cameras."""
        for cam in self._camera_array:
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(self._fps)
            logger.info(
                f"Cam {cam.GetCameraContext()} FPS set to {cam.AcquisitionFrameRate.Value}, "
                f"resulting FPS: {cam.ResultingFrameRate.Value}"
            )

    def disable_throughput_limit(self) -> None:
        """Turn off the device link throughput limit on all cameras."""
        for cam in self._camera_array:
            cam.DeviceLinkThroughputLimitMode.SetValue("Off")

    def log_device_link_info(self) -> None:
        """Log current device link throughput and speed for all cameras."""
        for cam in self._camera_array:
            logger.info(
                f"Cam {cam.GetCameraContext()} — "
                f"current throughput: {cam.DeviceLinkCurrentThroughput.Value}, "
                f"link speed: {cam.DeviceLinkSpeed.Value}, "
                f"throughput limit: {cam.DeviceLinkThroughputLimit.Value}"
            )

    # ------------------------------------------------------------------
    # Core grab loop (private)
    # ------------------------------------------------------------------

    def _run(self, condition: Callable[[list[int]], bool], max_frames: int) -> None:
        """
        Core grab loop. Runs until condition(frame_counts) returns True.

        Args:
            condition: Called each frame with the current per-camera frame counts.
                       Loop exits when it returns True.
            max_frames: Used to pre-allocate the timestamps array.
        """
        frame_counts = [0] * self._n_cameras
        timestamps = np.zeros((self._n_cameras, max_frames), dtype=np.int64)

        self.disable_throughput_limit()
        starting_mapping = latch_timestamp_mapping(self._camera_array)
        self.disable_throughput_limit()

        self._camera_array.StartGrabbing()
        self.set_fps_on_cameras()
        logger.info("Recording started.")

        try:
            while True:
                with self._camera_array.RetrieveResult(1000) as result:
                    if result.GrabSucceeded():
                        image_number = result.ImageNumber
                        cam_id = result.GetCameraContext()
                        frame_counts[cam_id] = image_number

                        relative_ts = (
                            result.GetTimeStamp()
                            - starting_mapping.camera_timestamps[cam_id]
                        )
                        logger.debug(f"Cam #{cam_id}  frame #{image_number}  ts: {relative_ts}")

                        try:
                            timestamps[cam_id, image_number - 1] = relative_ts
                        except IndexError:
                            pass  # Exceeded pre-allocated buffer; trimmed on save.

                        self._writer.write_frame_ffmpeg(frame=result.Array, cam_id=cam_id)

                        if condition(frame_counts):
                            break
                    else:
                        logger.error(
                            f"Grab failed — cam {result.GetCameraContext()}: "
                            f"{result.GetErrorDescription()} (ts={result.GetTimeStamp()})"
                        )
        finally:
            self._writer.release_ffmpeg()
            self._camera_array.StopGrabbing()

        ending_mapping = latch_timestamp_mapping(self._camera_array)
        save_timestamps(
            output_path=self._output_path,
            timestamps=timestamps,
            starting_mapping=starting_mapping,
            ending_mapping=ending_mapping,
        )
        logger.info(f"Frame counts: {frame_counts}")
        self.pylon_internal_statistics()
