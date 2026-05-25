"""Tests for reflections.env."""

from __future__ import annotations

import os
from pathlib import Path

from reflections.env import load_env


def test_load_env_overrides_existing(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'USER_NAME="Test User"\n' "# comment\n" "EMPTY=\n" "FLAG=true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("USER_NAME", "Old Name")
    load_env(env_file)
    assert os.environ["USER_NAME"] == "Test User"
    assert os.environ["FLAG"] == "true"


def test_load_env_missing_file_is_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NONEXISTENT_TEST_KEY", raising=False)
    load_env(tmp_path / "missing.env")
    assert "NONEXISTENT_TEST_KEY" not in os.environ
