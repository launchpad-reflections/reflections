"""Tests for proactivity.promptlog."""

from __future__ import annotations

import json

from proactivity import promptlog


def test_log_event_appends_jsonl(tmp_repo) -> None:
    promptlog.reset_log()
    promptlog.log_event("classifier", "prompt", {"target": "hello"})
    promptlog.log_event("classifier", "result", {"p": 0.42})

    lines = promptlog.LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["category"] == "classifier"
    assert first["stage"] == "prompt"
    assert first["target"] == "hello"
    assert first["seq"] == 1


def test_block_to_dict_from_attributes() -> None:
    class Block:
        type = "text"
        text = "hi"

    assert promptlog.block_to_dict(Block()) == {"type": "text", "text": "hi"}


def test_reset_log_truncates(tmp_repo) -> None:
    promptlog.log_event("test", "stage", {"x": 1})
    promptlog.reset_log()
    assert promptlog.LOG_PATH.read_text(encoding="utf-8") == ""
