"""Central env-driven configuration for Reflections."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# packages/reflections/config.py -> packages/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

APP_PORT = int(os.environ.get("PORT", "3000"))
SPEAK_URL = os.environ.get("SPEAK_URL", f"http://127.0.0.1:{APP_PORT}/speak")
TRANSCRIPT_URL = os.environ.get("TRANSCRIPT_URL", f"http://127.0.0.1:{APP_PORT}/transcripts")

WHEP_URL = os.environ.get("WHEP_URL", "http://127.0.0.1:8889/live/glasses/whep")

DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8766"))

SHOW_INTERIM = os.environ.get("SHOW_INTERIM", "false").lower() in ("1", "true", "yes")
TRANSCRIPT_LOG_PATH = os.environ.get("TRANSCRIPT_LOG_PATH", "transcript_updates.log")
USE_MENTRA_TRANSCRIPTION = os.environ.get("USE_MENTRA_TRANSCRIPTION", "false").lower() in (
    "1",
    "true",
    "yes",
)

DEFAULT_PHRASE = os.environ.get("DEFAULT_PHRASE", "Reflections speaker check")

SONIOX_API_KEY = os.environ.get("SONIOX_API_KEY", "")
USER_NAME = os.environ.get("USER_NAME", "User") or "User"

PROACTIVITY_ENABLED = os.environ.get("PROACTIVITY_ENABLED", "1").lower() not in (
    "0",
    "false",
    "no",
)

# Default location for places/directions tools. The shipped defaults are
# intentionally non-geographic ("Example City" at 0,0): real Maps lookups
# only return useful results once the user overrides these in `.env`.
DEFAULT_LOCATION_NAME = os.environ.get("DEFAULT_LOCATION_NAME", "Example City, CA")
DEFAULT_LOCATION_LAT = float(os.environ.get("DEFAULT_LOCATION_LAT", "0.0"))
DEFAULT_LOCATION_LON = float(os.environ.get("DEFAULT_LOCATION_LON", "0.0"))
DEFAULT_LOCATION_RADIUS_M = int(os.environ.get("DEFAULT_LOCATION_RADIUS_M", "5000"))

GLASSES_GATE_THRESHOLD = float(os.environ.get("GLASSES_GATE_THRESHOLD", "0.25"))

LORA_MODEL_ID = os.environ.get(
    "REFLECTIONS_LORA_MODEL_ID", "rushilsaraf/qwen3-actionable-v2-adapter"
)


def default_location() -> dict[str, Any]:
    """Location blob shared by the classifier and maps tools."""
    return {
        "description": f"Indoor, {DEFAULT_LOCATION_NAME}",
        "coordinates": {"latitude": DEFAULT_LOCATION_LAT, "longitude": DEFAULT_LOCATION_LON},
        "nearby_places": [],
    }
