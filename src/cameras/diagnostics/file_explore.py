import subprocess
import cv2
import numpy as np

from pathlib import Path
from datetime import datetime

from skellycam_plots import create_timestamp_diagnostic_plots, timestamps_array_to_dictionary
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)

Z_SCORE_95_CI = 1.96

# TODO: Make plots of timestamps for each camera

def print_video_info(folder_path: Path):
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)
    if not folder_path.exists:
        raise FileNotFoundError("Input folder path does not exist")

    raw_videos_path = folder_path / "raw_videos"
    if not raw_videos_path.exists():
        raw_videos_path = folder_path

    synched_videos_path = folder_path / "synchronized_videos"

    logger.info("Raw Video Information:")
    print_basic_info(raw_videos_path)

    if synched_videos_path.exists():
        logger.info("Synchronized Video Information:")
        print_basic_info(synched_videos_path)


    print_timestamp_info(raw_video_path = raw_videos_path, synched_video_path=synched_videos_path)

def print_basic_info(folder_path: Path):
    for video_path in folder_path.iterdir():
        if video_path.suffix not in {".avi", ".AVI", ".mp4", ".MP4"}:
            continue
        logger.info("  video name: %s", video_path.name)
        cap = cv2.VideoCapture(str(video_path))
        logger.info("  frame count: %s", cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("  reported fps: %s", cap.get(cv2.CAP_PROP_FPS))
        ffprobe_fps = get_ffprobe_fps(
            video_path
        )
        logger.info("  ffprobe fps: %s", ffprobe_fps)
        cap.release()

def print_timestamp_info(raw_video_path: Path, synched_video_path: Path):
    timestamp_file_name = "timestamps.npy"
    raw_timestamps_path = raw_video_path / timestamp_file_name
    synched_timestamps_path = synched_video_path / timestamp_file_name
    if raw_timestamps_path.exists():
        raw_timestamps = np.load(raw_timestamps_path) 
        print_timestamp_statistics(timestamps=raw_timestamps)
        raw_timestamp_dict = timestamps_array_to_dictionary(raw_timestamps)

    if synched_timestamps_path.exists():
        synched_timestamps = np.load(synched_timestamps_path)
        print_timestamp_statistics(timestamps=synched_timestamps)
        synched_timestamp_dict = timestamps_array_to_dictionary(synched_timestamps)

    if raw_timestamps_path.exists():
        if not synched_timestamps_path.exists():
            synched_timestamp_dict = None
        logger.info("Creating timestamp diagnostic plots, will save to: %s", synched_video_path.parent / "timestamp_diagnostic_plot.png")
        create_timestamp_diagnostic_plots(
            raw_timestamp_dictionary=raw_timestamp_dict,
            synchronized_timestamp_dictionary=synched_timestamp_dict,
            path_to_save_plots_png=synched_video_path.parent / "timestamp_diagnostic_plot.png"
        )

def print_timestamp_statistics(timestamps: np.ndarray):
    logger.info("shape of timestamps: %s", timestamps.shape)
    starting_time = np.min(timestamps)

    by_camera_fps = []
    by_camera_frame_duration = []

    for i in range(timestamps.shape[0]):
        num_samples = timestamps.shape[1]
        samples = (timestamps[i, :] - starting_time) / 1e9
        fps = num_samples / (samples[-1] - samples[0])
        mean_frame_duration = np.mean(np.diff(timestamps[i, :])) / 1e6
        by_camera_fps.append(fps)
        by_camera_frame_duration.append(mean_frame_duration)
        units = "seconds"
        logger.info("cam %d Descriptive Statistics:", i)
        logger.debug("  Earliest Timestamp: %.3f %s", np.min(samples), units)
        logger.debug("  Latest Timestamp:   %.3f %s", np.max(samples), units)
        logger.info("  FPS: %s", fps)
        logger.info("  Mean Frame Duration: %s ms", mean_frame_duration)

    logger.info("Overall FPS and Mean Frame Duration")
    logger.info("  Mean Overall FPS: %s", np.nanmean(by_camera_fps))
    logger.info("  Mean Overall Mean Frame Duration: %s", np.nanmean(by_camera_frame_duration))

    for i in range(0, timestamps.shape[1]-1, 15):
        num_samples = timestamps.shape[0]
        samples = (timestamps[:, i] - starting_time) / 1e9
        units = "seconds"
        logger.debug("frame %d Descriptive Statistics", i)
        logger.debug("  Number of Samples: %d %s", num_samples, units)
        logger.debug("  Mean:   %.3f %s", np.nanmean(samples), units)
        logger.debug("  Median: %.3f %s", np.nanmedian(samples), units)
        logger.debug("  Std Dev: %.3f %s", np.nanstd(samples), units)
        logger.debug("  Median Absolute Deviation: %.3f %s", np.nanmedian(np.abs(samples - np.nanmedian(samples))), units)
        logger.debug("  IQR: %.3f %s", np.nanpercentile(samples, 75) - np.nanpercentile(samples, 25), units)
        logger.debug("  95%% CI: %.3f %s", Z_SCORE_95_CI * np.nanstd(samples) / (num_samples**0.5), units)
        logger.debug("  Earliest Timestamp: %.3f", np.min(samples))
        logger.debug("  Latest Timestamp:   %.3f", np.max(samples))
        logger.debug("  Mean Frame Duration: %s ms", np.nanmean(timestamps[:, i+1] - timestamps[:, i]) / 1e6)


def get_ffprobe_fps(video_path: Path) -> float:
    duration_subprocess = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            f"{video_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = duration_subprocess.stdout
    logger.debug("duration from ffprobe: %s seconds", float(duration))

    frame_count_subprocess = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            f"{video_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    frame_count = int(frame_count_subprocess.stdout)

    logger.debug("frame count from ffprobe: %d", frame_count)

    return frame_count / float(duration)


if __name__ == "__main__":
    folder_path = Path(
        "/home/scholl-lab/recordings/test__3"
    )

    print_video_info(folder_path)

