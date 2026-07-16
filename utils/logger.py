"""
utils/logger.py — Centralized logging setup.
Call get_logger(__name__) at the top of any module.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from config import Config


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with both console and rotating file handlers.
    Safe to call multiple times with the same name — returns cached logger.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler — DEBUG and above, rotates at 5MB, keeps 3 backups
    Config.LOGS_DIR.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger