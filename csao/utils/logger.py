import logging
import os
from pathlib import Path
from typing import Optional


def get_logger(name: str, logs_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with console (and optional file) handlers.

    Parameters
    ----------
    name:
        Logger name (usually __name__ of the calling module).
    logs_dir:
        Optional directory to write log files into. If provided, a rotating
        file handler is attached. The directory will be created if needed.
    level:
        Logging level (e.g. logging.INFO).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if logs_dir is not None:
        path = Path(logs_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

