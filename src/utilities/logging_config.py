"""
Shared logging configuration for the bs pipeline.

All modules should call get_logger(__name__) to get a named logger.
The root pipeline logger is configured once (console + optional file) on first
call; subsequent calls just return the named child logger.

Usage:
    from src.utilities.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Starting step %s", step_name)

Log levels:
    DEBUG   — fine-grained trace info (file only by default)
    INFO    — normal operational messages
    WARNING — something unexpected but recoverable
    ERROR   — a step failed; pipeline may still continue
"""

import logging
from datetime import datetime
from pathlib import Path

# Root logger name shared across the whole pipeline.
_ROOT_LOGGER_NAME = "bs"

# Default directory for log files. Falls back silently if unavailable.
_DEFAULT_LOG_DIR = Path("/home/scholl-lab/pipeline_logs")

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Tracks whether the root logger has been configured so we only add handlers once.
_root_configured = False


def _configure_root_logger(
    log_dir: Path,
    file_level: int,
    console_level: int,
) -> None:
    """Add console and optional file handlers to the root pipeline logger."""
    global _root_configured
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)

    if root.handlers:
        _root_configured = True
        return

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional rotating file handler — skipped silently if the directory is not writable.
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(log_dir / f"pipeline_{timestamp}.log")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.debug("Could not create log file in %s — file logging disabled.", log_dir)

    _root_configured = True


def get_logger(
    name: str,
    log_dir: Path | None = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a named logger under the shared pipeline root.

    Configures the root logger on first call. Subsequent calls with any name
    reuse the existing handlers — so log_dir / level args only take effect on
    the very first call in a process.

    Args:
        name: Module name, typically __name__.
        log_dir: Where to write the log file. Defaults to _DEFAULT_LOG_DIR.
        file_level: Minimum level written to the log file.
        console_level: Minimum level written to the console.

    Returns:
        A logging.Logger instance named "<root>.<name>".
    """
    if not _root_configured:
        _configure_root_logger(
            log_dir=log_dir or _DEFAULT_LOG_DIR,
            file_level=file_level,
            console_level=console_level,
        )

    # Build a child logger so messages carry the full module path.
    child_name = f"{_ROOT_LOGGER_NAME}.{name}" if not name.startswith(_ROOT_LOGGER_NAME) else name
    return logging.getLogger(child_name)
