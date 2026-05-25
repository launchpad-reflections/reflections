"""Append-only JSONL event log for the live debugging dashboard.

Every prompt sent to the Qwen classifier and to Claude — plus every
client-side tool call — gets one JSON line in
proactivity_prompts.jsonl. proactivity/dashboard.html polls this file
and renders new lines as cards.

Thread-safe: one Lock serializes appends so the worker thread and the
main thread can't tear lines.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from reflections.config import REPO_ROOT

LOG_PATH = REPO_ROOT / "proactivity_prompts.jsonl"

_lock = threading.Lock()
_seq = 0


def _next_seq() -> int:
    global _seq
    _seq += 1
    return _seq


def log_event(category: str, stage: str, data: dict[str, Any]) -> None:
    """Append one JSON line. Never raises — logging must never perturb
    the main pipeline."""
    record = {
        "seq": _next_seq(),
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "category": category,
        "stage": stage,
        **data,
    }
    line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
    try:
        with _lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def reset_log() -> None:
    """Truncate the log. Useful before a fresh session."""
    try:
        with _lock:
            LOG_PATH.write_text("", encoding="utf-8")
    except Exception:
        pass


def block_to_dict(block: Any) -> dict[str, Any]:
    """Best-effort serialization of an Anthropic SDK content block to a
    dict. Falls back to attribute scraping if model_dump isn't there."""
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    out: dict[str, Any] = {"type": getattr(block, "type", "?")}
    for attr in ("text", "name", "input", "id", "tool_use_id", "content"):
        v = getattr(block, attr, None)
        if v is not None:
            out[attr] = v
    return out
