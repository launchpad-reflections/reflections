"""Central logging setup for Reflections."""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """Configure root logging from ``LOG_LEVEL`` (default ``INFO``)."""
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
