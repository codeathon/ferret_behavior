import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def plot_timestamps(timestamps: np.ndarray, save_path: Path | None = None):
    num_cameras = timestamps.shape[0]

    fig = plt.figure(figsize=(18,12))

    fig.suptitle(f"Timestamps")

    ax = plt.subplot()

    for i in range(num_cameras):
        
        ax.plot(timestamps[i], label=f"Camera {i}")
        ax.legend()

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(str(save_path))

        plt.show()

if __name__ == "__main__":
    timestamp_path = Path("/home/scholl-lab/recordings/session_2025-07-28/test__4/raw_videos/timestamps.npy")
    timestamps = np.load(timestamp_path)

    timestamps /= 1e9

    logger.info("timestamps shape: %s", timestamps.shape)

    plot_timestamps(timestamps=timestamps, save_path = timestamp_path.parent / "timestamp_plot.png")