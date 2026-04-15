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

from src.utilities.logging_config import get_logger
import queue
import threading
import time
from typing import Callable

import numpy as np
import pypylon.pylon as pylon

from src.cameras.synchronization.realtime_sync import (
    BaslerFrame,
    BaslerFrameSet,
    BaslerFrameSetCombiner,
)
from src.cameras.timestamp_utils import (
    basler_frame_utc_ns_from_latch_delta,
    latch_timestamp_mapping,
    save_timestamps,
)
from src.cameras.video_writers import VideoWriterManager

logger = get_logger(__name__)

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
        frameset_sink: If set, invoked on the combiner thread for each emitted
            ``BaslerFrameSet``. When non-None, frame payloads include a BGR uint8
            image copy (see ``BaslerFrame.payload``); keep the callback fast and
            thread-safe.
    """

    def __init__(
        self,
        camera_array: pylon.InstantCameraArray,
        n_cameras: int,
        fps: float,
        writer: VideoWriterManager,
        output_path,
        ring_size: int = 240,
        combiner_tolerance_ns: int = 2_000_000,
        frameset_sink: Callable[[BaslerFrameSet], None] | None = None,
    ) -> None:
        self._camera_array = camera_array
        self._n_cameras = n_cameras
        self._fps = fps
        self._writer = writer
        self._output_path = output_path
        self._ring_size = ring_size
        self._combiner_tolerance_ns = combiner_tolerance_ns
        self._frameset_sink = frameset_sink

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
        frameset_count = 0

        combiner = BaslerFrameSetCombiner(
            camera_ids=list(range(self._n_cameras)),
            ring_size=self._ring_size,
            tolerance_ns=self._combiner_tolerance_ns,
        )
        frame_queue: queue.Queue[BaslerFrame | None] = queue.Queue(maxsize=max(2_048, self._n_cameras * 128))
        combiner_stop = threading.Event()
        combiner_error: list[Exception] = []

        def _combiner_worker() -> None:
            nonlocal frameset_count
            while not combiner_stop.is_set():
                try:
                    item = frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                try:
                    frameset = combiner.ingest(item)
                    if frameset is not None:
                        frameset_count += 1
                        if self._frameset_sink is not None:
                            self._frameset_sink(frameset)
                except Exception as exc:  # defensive: avoid silent thread death
                    combiner_error.append(exc)
                    combiner_stop.set()
                    break

        combiner_thread = threading.Thread(
            target=_combiner_worker,
            name="basler-frameset-combiner",
            daemon=True,
        )
        combiner_thread.start()

        self.disable_throughput_limit()
        starting_mapping = latch_timestamp_mapping(self._camera_array)
        self.disable_throughput_limit()
        grab_anchor_utc_ns = starting_mapping.utc_time_ns

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

                        device_ts = int(result.GetTimeStamp())
                        latched = int(starting_mapping.camera_timestamps[cam_id])
                        relative_ts = device_ts - latched
                        logger.debug(f"Cam #{cam_id}  frame #{image_number}  ts: {relative_ts}")

                        try:
                            timestamps[cam_id, image_number - 1] = relative_ts
                        except IndexError:
                            pass  # Exceeded pre-allocated buffer; trimmed on save.

                        utc_ns = basler_frame_utc_ns_from_latch_delta(
                            device_timestamp=device_ts,
                            latched_device_timestamp=latched,
                            grab_anchor_utc_ns=grab_anchor_utc_ns,
                        )
                        payload = None
                        if self._frameset_sink is not None:
                            payload = np.ascontiguousarray(
                                np.asarray(result.Array, dtype=np.uint8)
                            )

                        try:
                            frame_queue.put_nowait(
                                BaslerFrame(
                                    camera_id=cam_id,
                                    frame_index=image_number,
                                    capture_utc_ns=utc_ns,
                                    payload=payload,
                                )
                            )
                        except queue.Full:
                            logger.warning("Combiner queue full; dropping combiner frame for cam %d", cam_id)

                        self._writer.write_frame_ffmpeg(frame=result.Array, cam_id=cam_id)

                        if combiner_error:
                            raise RuntimeError("Frame combiner thread failed") from combiner_error[0]

                        if condition(frame_counts):
                            break
                    else:
                        logger.error(
                            f"Grab failed — cam {result.GetCameraContext()}: "
                            f"{result.GetErrorDescription()} (ts={result.GetTimeStamp()})"
                        )
        finally:
            combiner_stop.set()
            try:
                frame_queue.put_nowait(None)
            except queue.Full:
                pass
            combiner_thread.join(timeout=2.0)
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
        logger.info("Near-synchronous Basler frame-sets emitted: %d", frameset_count)
        self.pylon_internal_statistics()
