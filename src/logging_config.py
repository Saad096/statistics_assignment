# src/logging_config.py
import logging
from pathlib import Path


def get_logger(
    name: str = "homework4", log_path: str = "outputs/logs/homework4.log"
) -> logging.Logger:
    """
    Create or get a logger that logs to both console and a file.
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        ch.setFormatter(ch_formatter)

        # File handler
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fh_formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
