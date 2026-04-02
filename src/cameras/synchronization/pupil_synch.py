import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.utilities.logging_config import get_logger
from src.cameras.synchronization.time_units import (
    seconds_to_nanoseconds,
    nanoseconds_to_seconds,
    resolve_synchronized_video_dir,
)

logger = get_logger(__name__)


class PupilSynchronize:
    def __init__(self, folder_path: Path):
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError("Input folder path does not exist")

        self.raw_videos_path = folder_path / "raw_videos"
        self.synched_videos_path = resolve_synchronized_video_dir(folder_path)
        self.output_path = folder_path / "basler_pupil_synchronized"

        self.basler_timestamp_mapping_file_name = "timestamp_mapping.json"
        basler_timestamp_mapping_file = (
                self.raw_videos_path / self.basler_timestamp_mapping_file_name
        )
        with open(basler_timestamp_mapping_file) as basler_timestamp_mapping_file:
            self.basler_timestamp_mapping = json.load(basler_timestamp_mapping_file)

        self.pupil_path = folder_path / "pupil_output"
        self.pupil_eye0_video_path = self.pupil_path / "eye0.mp4"
        self.pupil_eye1_video_path = self.pupil_path / "eye1.mp4"

        self.pupil_timestamp_mapping_file_name = "info.player.json"
        pupil_timestamp_mapping_file = (
                self.pupil_path / self.pupil_timestamp_mapping_file_name
        )
        with open(pupil_timestamp_mapping_file) as pupil_timestamp_mapping_file:
            self.pupil_timestamp_mapping = json.load(pupil_timestamp_mapping_file)

        self.synchronization_metadata = {}

        self.load_pupil_timestamps()
        self.load_basler_timestamps()
        self.load_index_to_serial_number()
        self.verify_index_to_serial_number()

    def seconds_to_nanoseconds(self, seconds: float) -> int:
        return seconds_to_nanoseconds(seconds)

    def nanoseconds_to_seconds(self, nanoseconds: int) -> float:
        return nanoseconds_to_seconds(nanoseconds)

    @property
    def pupil_start_time(self) -> int:
        return self.seconds_to_nanoseconds(
            self.pupil_timestamp_mapping["start_time_synced_s"]
        )

    @property
    def pupil_start_time_utc(self) -> int:
        return self.seconds_to_nanoseconds(
            self.pupil_timestamp_mapping["start_time_system_s"]
        )

    @property
    def pupil_first_synched_timestamp_utc(self) -> int:
        return int(
            min(
                np.min(self.pupil_eye0_timestamps_utc),
                np.min(self.pupil_eye1_timestamps_utc),
            )
        )

    @property
    def pupil_last_synched_timestamp_utc(self) -> int:
        return int(
            min(
                np.max(self.pupil_eye0_timestamps_utc),
                np.max(self.pupil_eye1_timestamps_utc),
            )
        )

    @property
    def basler_start_time(self) -> int:
        return self.basler_timestamp_mapping["starting_mapping"]["perf_counter_ns"]

    @property
    def basler_start_time_utc(self) -> int:
        return self.basler_timestamp_mapping["starting_mapping"]["utc_time_ns"]

    @property
    def basler_first_synched_timestamp(self) -> int:
        return int(np.min(self.synched_basler_timestamps))

    @property
    def basler_first_synched_timestamp_utc(self) -> int:
        return int(np.min(self.synched_basler_timestamps_utc))

    @property
    def difference_in_start_times(self) -> int:
        return self.basler_start_time_utc - self.pupil_start_time_utc

    @property
    def basler_end_time(self) -> int:
        return self.basler_timestamp_mapping["ending_mapping"]["perf_counter_ns"]

    @property
    def basler_end_time_utc(self) -> int:
        return self.basler_timestamp_mapping["ending_mapping"]["utc_time_ns"]

    @property
    def basler_last_synched_timestamp(self) -> int:
        return int(np.min(self.synched_basler_timestamps[:, -1]))

    @property
    def basler_last_synched_timestamp_utc(self) -> int:
        return int(np.min(self.synched_basler_timestamps_utc[:, -1]))

    @property
    def length_of_basler_recording(self) -> int:
        return self.basler_last_synched_timestamp - self.basler_first_synched_timestamp

    @property
    def latest_synched_start_utc(self) -> int:
        return max(
            self.pupil_first_synched_timestamp_utc,
            self.basler_first_synched_timestamp_utc,
        )

    @property
    def earliest_synched_end_utc(self) -> int:
        return min(
            self.pupil_last_synched_timestamp_utc,
            self.basler_last_synched_timestamp_utc,
        )

    @property
    def basler_camera_names(self) -> List[str]:
        return list(
            self.basler_timestamp_mapping["starting_mapping"][
                "camera_timestamps"
            ].keys()
        )

    @property
    def pupil_camera_names(self) -> List[str]:
        return ["eye0", "eye1"]

    def load_pupil_timestamps(self):
        """Load pupil timestamps and convert to utc"""
        pupil_eye0_timestamps_path = self.pupil_path / "eye0_timestamps.npy"
        pupil_eye0_timestamps = np.load(pupil_eye0_timestamps_path)
        pupil_eye0_timestamps *= 1e9  # convert to ns
        pupil_eye0_timestamps = pupil_eye0_timestamps.astype(int)  # cast to int
        pupil_eye0_timestamps -= (
            self.pupil_start_time
        )  # convert to ns since pupil start time
        self.pupil_eye0_timestamps_utc = (
                pupil_eye0_timestamps + self.pupil_start_time_utc
        )

        pupil_eye1_timestamps_path = self.pupil_path / "eye1_timestamps.npy"
        pupil_eye1_timestamps = np.load(pupil_eye1_timestamps_path)
        pupil_eye1_timestamps *= 1e9  # convert to ns
        pupil_eye1_timestamps = pupil_eye1_timestamps.astype(int)  # cast to int
        pupil_eye1_timestamps -= (
            self.pupil_start_time
        )  # convert to ns since pupil start time
        self.pupil_eye1_timestamps_utc = (
                pupil_eye1_timestamps + self.pupil_start_time_utc
        )

    def load_basler_timestamps(self):
        """
        Load basler timestamps and convert to utc
        Basler timestamps are saved in ns since camera latch time, which is roughly equivalent to time since utc start
        """
        self.basler_timestamp_file_name = "timestamps.npy"
        synched_basler_timestamp_path = (
                self.synched_videos_path / self.basler_timestamp_file_name
        )
        self.synched_basler_timestamps = np.load(synched_basler_timestamp_path)
        self.synched_basler_timestamps_utc = (
                self.synched_basler_timestamps + self.basler_start_time_utc
        )

    def load_index_to_serial_number(self):
        index_to_serial_number_path = (
                self.raw_videos_path / "index_to_serial_number_mapping.json"
        )
        if not index_to_serial_number_path.exists():
            logger.warning("index_to_serial_number_mapping.json not found at %s — using default mapping", index_to_serial_number_path)
            logger.warning("Verify default mapping correctness before proceeding")
            self.index_to_serial_number = {
                "0": "24908831",
                "1": "24908832",
                "2": "25000609",
                "3": "25006505"
            }
        else:
            with open(index_to_serial_number_path) as index_to_serial_number_file:
                self.index_to_serial_number = json.load(index_to_serial_number_file)

        self.synchronization_metadata["index_to_serial_number_mapping"] = self.index_to_serial_number

    def verify_index_to_serial_number(self):
        logger.debug("Index-to-serial mapping (smaller serial numbers come first):")
        for cam_name in self.basler_camera_names:
            logger.debug("  cam %s serial: %s", cam_name, self.index_to_serial_number[cam_name])

    def get_pupil_fps(self) -> Tuple[float, float]:
        eye0_time_elapsed_s = (
                                      self.pupil_eye0_timestamps_utc[-1] - self.pupil_eye0_timestamps_utc[0]
                              ) / 1e9
        eye0_fps = self.pupil_eye0_timestamps_utc.shape[0] / eye0_time_elapsed_s

        eye1_time_elapsed_s = (
                                      self.pupil_eye1_timestamps_utc[-1] - self.pupil_eye1_timestamps_utc[0]
                              ) / 1e9
        eye1_fps = self.pupil_eye1_timestamps_utc.shape[0] / eye1_time_elapsed_s

        logger.info("Pupil actual fps — eye0: %.2f, eye1: %.2f", eye0_fps, eye1_fps)
        logger.debug("Average frame duration — eye0: %.1f ns, eye1: %.1f ns", 1 / eye0_fps * 1e9, 1 / eye1_fps * 1e9)
        logger.debug("FPS difference: %.4f", eye0_fps - eye1_fps)

        self.synchronization_metadata["pupil_input_fps"] = {
            "eye0": eye0_fps,
            "eye1": eye1_fps
        }

        return eye0_fps, eye1_fps

    def get_pupil_median_fps(self) -> Tuple[float, float]:
        eye_0_time_difference = np.diff(self.pupil_eye0_timestamps_utc)
        eye_0_fps = 1e9 / np.median(eye_0_time_difference)

        eye_1_time_difference = np.diff(self.pupil_eye1_timestamps_utc)
        eye_1_fps = 1e9 / np.median(eye_1_time_difference)

        logger.info("Pupil median fps — eye0: %.2f, eye1: %.2f", eye_0_fps, eye_1_fps)

        if eye_0_fps != eye_1_fps:
            logger.warning("Pupil median fps mismatch — eye0: %.2f vs eye1: %.2f", eye_0_fps, eye_1_fps)

        self.synchronization_metadata["pupil_median_fps"] = {
            "eye0": eye_0_fps,
            "eye1": eye_1_fps
        }

        return float(eye_0_fps), float(eye_1_fps)

    def get_utc_timestamp_per_camera(self) -> Dict[int, int]:
        return {
            int(camera): (self.basler_start_time_utc - basler_timestamp)
            for camera, basler_timestamp in self.basler_timestamp_mapping[
                "starting_mapping"
            ]["camera_timestamps"].items()
        }
    
    def get_closest_pupil_frame_to_basler_frame(self, basler_frame_number: int) -> tuple[int, int]:
        basler_utc = np.median(self.synched_basler_timestamps_utc[:, basler_frame_number])
        logger.debug("Basler UTC reference: %s", basler_utc)
        eye0_match = np.searchsorted(self.pupil_eye0_timestamps_utc, basler_utc, side="right")
        if (basler_utc - self.pupil_eye0_timestamps_utc[eye0_match-1]) < abs(basler_utc - self.pupil_eye0_timestamps_utc[eye0_match]):
            eye0_frame_number = eye0_match-1
        else: 
            eye0_frame_number = eye0_match

        eye1_match = np.searchsorted(self.pupil_eye0_timestamps_utc, basler_utc, side="right")
        if (basler_utc - self.pupil_eye1_timestamps_utc[eye1_match-1]) < abs(basler_utc - self.pupil_eye1_timestamps_utc[eye1_match]):
            eye1_frame_number = eye1_match-1
        else: 
            eye1_frame_number = eye1_match

        logger.debug("Eye 0 match: frame %d at utc %s", eye0_frame_number, self.pupil_eye0_timestamps_utc[eye0_frame_number])
        logger.debug("Eye 1 match: frame %d at utc %s", eye1_frame_number, self.pupil_eye1_timestamps_utc[eye1_frame_number])

        return eye0_frame_number, eye1_frame_number


    def find_starting_offsets_in_frames(self) -> Dict[str, int]:
        starting_offsets_in_frames = {
            cam_name: np.where(
                self.synched_basler_timestamps_utc[i, :] >= self.latest_synched_start_utc
            )[0][0]
            for i, cam_name in enumerate(self.basler_camera_names)
        }

        starting_offsets_in_frames["eye0"] = np.where(
            self.pupil_eye0_timestamps_utc >= self.latest_synched_start_utc
        )[0][0]
        starting_offsets_in_frames["eye1"] = np.where(
            self.pupil_eye1_timestamps_utc >= self.latest_synched_start_utc
        )[0][0]

        logger.debug("Starting offsets in frames: %s", starting_offsets_in_frames)
        return starting_offsets_in_frames

    def find_ending_offsets_in_frames(self) -> Dict[str, int]:
        ending_offsets_in_frames = {
            cam_name: np.where(
                self.synched_basler_timestamps_utc[i, :] >= self.earliest_synched_end_utc
            )[0][0]
            for i, cam_name in enumerate(self.basler_camera_names)
        }

        ending_offsets_in_frames["eye0"] = np.where(
            self.pupil_eye0_timestamps_utc >= self.earliest_synched_end_utc
        )[0][0]
        ending_offsets_in_frames["eye1"] = np.where(
            self.pupil_eye1_timestamps_utc >= self.earliest_synched_end_utc
        )[0][0]

        logger.debug("Ending offsets in frames: %s", ending_offsets_in_frames)
        return ending_offsets_in_frames

    def save_synchronized_timestamps(self):
        if self.synchronized_timestamps is None:
            raise ValueError(
                "synchronized_timestamps is None, this method should only be called from synchronize(), it should not be called directly")
        for cam_name, timestamps in self.synchronized_timestamps.items():
            logger.debug("cam %s timestamps shape: %s", cam_name, timestamps.shape)
            np.save(f"{self.output_path}/cam_{cam_name}_synchronized_timestamps.npy", timestamps)

    def save_metadata(self):
        with open(f"{self.output_path}/metadata.json", "w") as f:
            json.dump(self.synchronization_metadata, f, indent=4)

    def trim_single_video(self,
                          start_frame: int,
                          end_frame: int,
                          input_video_pathstring: str,
                          output_video_pathstring: str,
                          ):
        frame_list = list(range(start_frame, end_frame + 1))
        cap = cv2.VideoCapture(input_video_pathstring)

        framerate = cap.get(cv2.CAP_PROP_FPS)
        framesize = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")

        video_writer_object = cv2.VideoWriter(
            output_video_pathstring, fourcc, framerate, framesize
        )

        logger.info("Saving synchronized video to %s", output_video_pathstring)

        current_frame = 0
        written_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_list:
                video_writer_object.write(frame)
                written_frames += 1

            if written_frames == len(frame_list):
                break

            current_frame += 1

        cap.release()
        video_writer_object.release()

    def trim_single_video_pupil(self,
                                        camera_name: str,
                                        input_video_pathstring: str,
                                        output_video_pathstring: str,
                                        ):
        cap = cv2.VideoCapture(input_video_pathstring)

        raw_timestamps = self.pupil_eye0_timestamps_utc.copy() if camera_name == "eye0" else self.pupil_eye1_timestamps_utc.copy()
        camera_median_fps = self.get_pupil_median_fps()[0] if camera_name == "eye0" else self.get_pupil_median_fps()[1]
        camera_median_fps = round(camera_median_fps, 2)
        logger.debug("Camera median fps: %.2f", camera_median_fps)
        median_duration = 1e9 / camera_median_fps

        framerate = cap.get(cv2.CAP_PROP_FPS)
        framesize = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # need to deal with higher frame rates

        video_writer_object = cv2.VideoWriter(
            output_video_pathstring, fourcc, camera_median_fps, framesize
        )

        logger.info("Saving synchronized pupil video to %s", output_video_pathstring)

        current_frame = 0
        written_frames = 0
        dropped_frames = 0
        skipped_frames = 0
        early_frames = 0
        synchronized_timestamps: List[int | None] = []
        previous_frame = np.zeros((framesize[1], framesize[0], 3), dtype=np.uint8)

        while True:
            reference_timestamp = self.latest_synched_start_utc + (written_frames * median_duration)
            if current_frame >= len(raw_timestamps):
                if reference_timestamp > self.earliest_synched_end_utc:
                    logger.debug("Reached target ending timestamp, exiting")
                    break
                else:
                    logger.warning("Ran out of frames — target: %s, actual final: %s, ref: %s",
                                   self.earliest_synched_end_utc, raw_timestamps[-1], reference_timestamp)
                    # TODO: We may want to fill in dummy frames here
                    break
            current_timestamp = raw_timestamps[current_frame]
            if reference_timestamp > self.earliest_synched_end_utc:
                # past the last synchronized time
                logger.debug("Reached target ending timestamp, exiting")
                break
            elif current_timestamp < self.latest_synched_start_utc - (0.5 * median_duration):
                # before the first synchronized time
                early_frames += 1
                current_frame += 1
                continue

            # if we make it past the if/elif, the current timestamp is between the start and end times
            if current_timestamp > (reference_timestamp + (0.5 * median_duration)):
                # current frame is too late, don't read it and fill in a dummy frame instead
                frame = cv2.drawMarker(previous_frame, (20, 20), (0, 0, 255), cv2.MARKER_STAR, 30, 1)
                video_writer_object.write(frame)
                synchronized_timestamps.append(None)
                written_frames += 1
                dropped_frames += 1
            elif current_timestamp < (reference_timestamp - (0.5 * median_duration)):
                # current frame is too early, read it and move to the next frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Unable to read frame %d", current_frame)
                    raise ValueError("Unable to read frame")
                    break
                previous_frame = frame
                skipped_frames += 1
                current_frame += 1
            else:
                # current frame is in the correct time window
                ret, frame = cap.read()
                if not ret:
                    logger.error("Unable to read frame %d", current_frame)
                    raise ValueError("Unable to read frame")
                    break
                previous_frame = frame

                video_writer_object.write(frame)
                synchronized_timestamps.append(current_timestamp)

                written_frames += 1
                current_frame += 1

        logger.info("Saved %s — %d frames, %d dropped, %d early, %d skipped",
                    output_video_pathstring, written_frames, dropped_frames, early_frames, skipped_frames)

        self.synchronized_timestamps[camera_name] = np.array(synchronized_timestamps)

        cap.release()
        video_writer_object.release()

    def trim_videos(self, starting_offsets_frames: Dict[str, int], ending_offsets_frames: Dict[str, int]):
        for cam_name in self.basler_camera_names:
            self.trim_single_video(
                starting_offsets_frames[cam_name],
                ending_offsets_frames[cam_name],
                str(self.synched_videos_path / f"{self.index_to_serial_number[cam_name]}.mp4"),
                str(self.output_path / f"{self.index_to_serial_number[cam_name]}.mp4"),
            )

        self.trim_single_video_pupil(
            camera_name="eye0",
            input_video_pathstring=str(self.pupil_eye0_video_path),
            output_video_pathstring=str(self.output_path / "eye0.mp4"),
        )

        self.trim_single_video_pupil(
            camera_name="eye1",
            input_video_pathstring=str(self.pupil_eye1_video_path),
            output_video_pathstring=str(self.output_path / "eye1.mp4"),
        )

    def verify_framecounts(self):
        for cam_name in self.synchronized_timestamps.keys():
            if cam_name.startswith("eye"):
                cap = cv2.VideoCapture(str(self.output_path / f"{cam_name}.mp4")) # THIS PATH IS WRONG
            else:
                cap = cv2.VideoCapture(str(self.output_path / f"{self.index_to_serial_number[cam_name]}.mp4"))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count != self.synchronized_timestamps[cam_name].shape[0]:
                logger.warning("Frame count mismatch for cam %s: video %d vs timestamps %d",
                               cam_name, frame_count, self.synchronized_timestamps[cam_name].shape[0])
            else:
                logger.debug("Frame count match for cam %s: %d", cam_name, frame_count)

            cap.release()

    def synchronize(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Latest synched start utc: %d", self.latest_synched_start_utc)
        logger.info("Earliest synched end utc: %d", self.earliest_synched_end_utc)
        starting_offsets_frames = self.find_starting_offsets_in_frames()
        ending_offsets_frames = self.find_ending_offsets_in_frames()

        self.synchronization_metadata = {
            "latest_synched_start_utc": self.latest_synched_start_utc,
            "earliest_synched_end_utc": self.earliest_synched_end_utc,
            "starting_offsets_frames": starting_offsets_frames,
            "ending_offsets_frames": ending_offsets_frames,
        }

        self.synchronized_timestamps = {
            cam_name: self.synched_basler_timestamps_utc[i, :][
                      starting_offsets_frames[cam_name]: ending_offsets_frames[cam_name] + 1
                      ] for i, cam_name in enumerate(self.basler_camera_names)
        }

        self.trim_videos(starting_offsets_frames=starting_offsets_frames, ending_offsets_frames=ending_offsets_frames)
        self.save_synchronized_timestamps()
        self.verify_framecounts()
        # TODO: verify all video files exist and are readable

        self.plot_raw_timestamps(
            starting_offsets=starting_offsets_frames,
            ending_offsets=ending_offsets_frames
        )

        self.plot_synchronized_timestamps()

    def plot_raw_timestamps(
            self,
            starting_offsets: Dict[str, int],
            ending_offsets: Dict[str, int],
    ):
        """plot some diagnostics to assess quality of camera sync"""
        # TODO: swap time and frame number, so x axis shows synching
        # opportunistic load of matplotlib to avoid startup time costs
        from matplotlib import pyplot as plt

        plt.set_loglevel("warning")

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"Timestamps")

        ax1 = plt.subplot(
            title="(Raw) Camera Frame Timestamp vs Frame#\n(Lines should have same slope)",
            xlabel="Frame#",
            ylabel="Timestamp (ns)",
        )

        for i, cam_name in enumerate(self.basler_camera_names):
            ax1.plot(
                self.synched_basler_timestamps_utc[i, :],
                label=f"basler {cam_name}",
            )
        ax1.plot(self.pupil_eye0_timestamps_utc, label="eye0")
        ax1.plot(self.pupil_eye1_timestamps_utc, label="eye1")

        for name in starting_offsets.keys():
            ax1.vlines(
                [starting_offsets[name], ending_offsets[name]],
                ymin=self.latest_synched_start_utc,
                ymax=self.earliest_synched_end_utc,
                label=f"{name} vlines",
            )

        ax1.legend()
        ax1.set_ylim(self.latest_synched_start_utc, self.earliest_synched_end_utc)

        plt.tight_layout()

        # plt.show()
        plt.savefig(str(self.output_path / "raw_timestamps.png"))

    def plot_synchronized_timestamps(self):
        """plot some diagnostics to assess quality of camera sync"""
        # TODO: swap time and frame number, so x axis shows synching
        # opportunistic load of matplotlib to avoid startup time costs
        from matplotlib import pyplot as plt

        plt.set_loglevel("warning")

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"Timestamps")

        ax1 = plt.subplot(
            title="(Synchronized) Camera Frame Timestamp vs Frame#\n(Lines should have same slope)",
            xlabel="Frame#",
            ylabel="Timestamp (ns)",
        )

        for name, timestamps in self.synchronized_timestamps.items():
            ax1.plot(timestamps, label=name)

        ax1.legend()
        ax1.set_ylim(self.latest_synched_start_utc, self.earliest_synched_end_utc)

        plt.tight_layout()

        # plt.show()
        plt.savefig(str(self.output_path / "synchronized_timestamps.png"))


if __name__ == "__main__":
    folder_path = Path(
        "/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/"
    )
    pupil_synchronize = PupilSynchronize(folder_path)

    utc_timestamp_per_camera = pupil_synchronize.get_utc_timestamp_per_camera()
    utc_start_time_pupil = pupil_synchronize.pupil_start_time_utc
    utc_start_time_basler = pupil_synchronize.basler_start_time_utc

    logger.info("Pupil start (pupil ns): %d", pupil_synchronize.pupil_start_time)
    logger.info("Pupil start (utc ns): %d", utc_start_time_pupil)
    logger.info("Basler start (utc ns): %d", utc_start_time_basler)
    logger.info("Basler start (Basler ns): %d", pupil_synchronize.basler_start_time)
    logger.info("Start time difference (basler - pupil) s: %.6f", pupil_synchronize.difference_in_start_times / 1e9)
    logger.info("Pupil start datetime: %s", np.datetime64(utc_start_time_pupil, 'ns'))
    logger.info("Basler start datetime: %s", np.datetime64(utc_start_time_basler, 'ns'))
    logger.info("Basler start times per camera: %s", pupil_synchronize.basler_timestamp_mapping['starting_mapping']['camera_timestamps'])
    logger.info("Pupil timestamp shapes — eye0: %s eye1: %s", pupil_synchronize.pupil_eye0_timestamps_utc.shape, pupil_synchronize.pupil_eye1_timestamps_utc.shape)
    logger.debug("Pupil timestamps (eye0): %s", pupil_synchronize.pupil_eye0_timestamps_utc)
    logger.debug("Pupil timestamps (eye1): %s", pupil_synchronize.pupil_eye1_timestamps_utc)

    pupil_synchronize.get_closest_pupil_frame_to_basler_frame(3377)
    pupil_synchronize.get_closest_pupil_frame_to_basler_frame(8754)

    # np.save(str(folder_path / "pupil_output" / "eye0_timestamps_utc.npy"), pupil_synchronize.pupil_eye0_timestamps_utc)
    # np.save(str(folder_path / "pupil_output" / "eye1_timestamps_utc.npy"), pupil_synchronize.pupil_eye1_timestamps_utc)
    # pupil_synchronize.synchronize()
