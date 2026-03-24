import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, level: str = None) -> logging.Logger:
    """Get a configured logger with the given name."""
    log_level = level or os.environ.get("AURORA_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


class AuroraLogger:
    """Structured logger for AURORA-VISION pipeline stages."""

    def __init__(self, component: str):
        self.logger = get_logger(f"aurora.{component}")
        self.component = component
        self._stage_times: dict = {}

    def stage_start(self, stage: str) -> None:
        self._stage_times[stage] = datetime.now()
        self.logger.info(f"[START] {stage}")

    def stage_end(self, stage: str) -> float:
        if stage in self._stage_times:
            elapsed = (datetime.now() - self._stage_times[stage]).total_seconds() * 1000
            self.logger.info(f"[END]   {stage} — {elapsed:.1f}ms")
            return elapsed
        return 0.0

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)
