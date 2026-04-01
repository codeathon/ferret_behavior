import os
import cv2
from pathlib import Path
from typing import List

from python_code.utilities.logging_config import get_logger

logger = get_logger(__name__)


def find_mp4_files(folder_path: str) -> List[Path]:
    """Find all MP4 files in the specified folder and its subfolders."""
    folder = Path(folder_path)
    return list(folder.glob("**/*.mp4"))


def clip_video(video_path: Path, output_path: Path, start_frame: int, end_frame: int) -> bool:
    """Clip a video to the specified frame range."""
    try:
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video %s", video_path)
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate frame range
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            logger.error("Invalid frame range (%d-%d) for video with %d frames", start_frame, end_frame, total_frames)
            cap.release()
            return False

        # Create video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Use mp4v codec
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps, 
            (width, height)
        )

        # Set the frame position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            current_frame += 1

        # Release resources
        cap.release()
        out.release()
        
        logger.info("Clipped %s → %s", video_path.name, output_path)
        return True
        
    except Exception as e:
        logger.error("Error processing %s: %s", video_path, e)
        return False






if __name__ == "__main__":
    
    START_FRAME = 10900  # Starting frame (inclusive)
    END_FRAME = 11800  # Ending frame (inclusive)
    INPUT_FOLDER = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-05-04\ferret_9C04_NoImplant_P41_E9\synchronized_videos"  # Folder containing MP4 files
    RECORDING_NAME= Path(INPUT_FOLDER).parent.name 
    OUTPUT_FOLDER = Path(INPUT_FOLDER).parent / f"clips/{RECORDING_NAME}_{START_FRAME}-{END_FRAME}"  # Output folder for clipped videos


    if END_FRAME < START_FRAME:
        logger.error("END_FRAME must be >= START_FRAME")
        raise ValueError("END_FRAME must be greater than or equal to START_FRAME")
    if not os.path.exists(INPUT_FOLDER):
        logger.error("Input folder does not exist: %s", INPUT_FOLDER)
        raise FileNotFoundError(f"Input folder {INPUT_FOLDER} does not exist")
    
    
    # Find all MP4 files
    mp4_files = find_mp4_files(INPUT_FOLDER)
    
    if not mp4_files:
        logger.error("No MP4 files found in %s", INPUT_FOLDER)
        raise FileNotFoundError(f"No MP4 files found in {INPUT_FOLDER}")

    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("Found %d MP4 files", len(mp4_files))
    
    # Process each video
    for video_path in mp4_files:
        output_path = output_dir / f"clipped_{video_path.name}"
        clip_video(video_path, output_path, START_FRAME, END_FRAME)
