"""
Logging configuration for the cameras module.

Provides get_camera_logger() which sets up both a rotating file handler
(if the log directory exists and is writable) and a console handler.
Falls back gracefully to console-only logging if the log directory is
unavailable, so imports never crash on machines without the lab path.
"""

import logging
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_DIR = Path("/home/scholl-lab/recordings/basler_logs")


def get_camera_logger(
    name: str,
    log_dir: Path | None = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Build and return a logger with a console handler and an optional file handler.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_dir: Directory for the log file. Defaults to DEFAULT_LOG_DIR.
                 If the directory does not exist or is not writable, file
                 logging is skipped silently.
        file_level: Log level for the file handler.
        console_level: Log level for the console handler.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    resolved_log_dir = log_dir or DEFAULT_LOG_DIR
    try:
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = resolved_log_dir / f"{timestamp}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.debug(f"Could not create log file in {resolved_log_dir} — file logging disabled.")

    return logger
