"""Session-scoped log of suggestions Claude has already proposed.

Layout (all under repo root):

    suggestions/
        current.md                       # the live session
        archive/
            2026-05-04_17-15-13.md       # rotated previous sessions

The agent's __init__ calls reset_suggestions() which:
  1. One-time-migrates the legacy repo-root `suggestions.md` into
     archive/ if it still exists.
  2. Rotates the prior session's `current.md` into archive/<timestamp>.md
     (only if it actually had content; an empty session is just deleted).
  3. Opens a fresh `current.md` for the new session.

Result: you keep a permanent record per session, never lose history,
and Claude's anti-repeat prompt only sees the current session's
suggestions (the archive is for you, not the model).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from reflections.config import REPO_ROOT

SUGGESTIONS_DIR = REPO_ROOT / "suggestions"
CURRENT_PATH = SUGGESTIONS_DIR / "current.md"
ARCHIVE_DIR = SUGGESTIONS_DIR / "archive"

# Old single-file location, kept just long enough to migrate any
# existing log into the new archive on first run.
_LEGACY_PATH = REPO_ROOT / "suggestions.md"

_HEADER = "# Session suggestions\n\n"

_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _has_content(path: Path) -> bool:
    """True if the file has at least one bullet line (not just header)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except Exception as e:
        _logger.debug("could not read suggestions file %s: %s", path, e)
        return False
    return any(ln.strip().startswith("- ") for ln in text.splitlines())


def _archive_file(path: Path) -> None:
    """Move `path` into suggestions/archive/ with a timestamped name.
    Adds a numeric suffix on collision (rapid sequential sessions)."""
    if not path.exists():
        return
    _ensure_dirs()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = ARCHIVE_DIR / f"{stamp}.md"
    suffix = 0
    while dest.exists():
        suffix += 1
        dest = ARCHIVE_DIR / f"{stamp}_{suffix}.md"
    path.rename(dest)


def reset_suggestions(path: Path = CURRENT_PATH) -> None:
    """Rotate the prior session's current.md into archive/ and open a
    fresh current.md. Called once by ProactivityAgent.__init__."""
    try:
        with _lock:
            _ensure_dirs()
            # One-time migration of the legacy repo-root file.
            if _LEGACY_PATH.exists():
                if _has_content(_LEGACY_PATH):
                    _archive_file(_LEGACY_PATH)
                else:
                    _LEGACY_PATH.unlink(missing_ok=True)
            # Rotate the prior current.md if it has anything in it,
            # otherwise just remove it cleanly.
            if path.exists():
                if _has_content(path):
                    _archive_file(path)
                else:
                    path.unlink(missing_ok=True)
            # Fresh current.md.
            path.write_text(_HEADER, encoding="utf-8")
    except Exception as e:
        _logger.debug("reset_suggestions failed: %s", e)


def append_suggestion(
    text: str,
    *,
    reason: str | None = None,
    tool_names: list[str] | None = None,
    path: Path = CURRENT_PATH,
) -> None:
    """Append one Claude-proposed suggestion to current.md. Captures
    the spoken text, Claude's stated reason, and any tools that fired
    so side-effecting actions (create_calendar_event etc.) surface in
    the log even if the agent's downstream gates suppressed speaking."""
    ts = datetime.now().strftime("%H:%M:%S")
    safe_text = text.replace("\n", " ").strip()
    parts = [f'- [{ts}] "{safe_text}"']
    if reason:
        safe_reason = reason.replace("\n", " ").strip()
        parts.append(f"_(reason: {safe_reason})_")
    if tool_names:
        parts.append(f"[tools: {', '.join(tool_names)}]")
    line = " ".join(parts) + "\n"
    try:
        with _lock:
            _ensure_dirs()
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        _logger.debug("append_suggestion failed: %s", e)


def read_suggestions(path: Path = CURRENT_PATH) -> str:
    """Return the bullet lines of the current session, stripped of
    header. Empty string if the file is missing or has no entries."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as e:
        _logger.debug("read_suggestions failed: %s", e)
        return ""
    bullets = [ln for ln in text.splitlines() if ln.strip().startswith("- ")]
    return "\n".join(bullets)
