"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point REPO_ROOT-backed modules at an isolated directory."""
    monkeypatch.setattr("reflections.config.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proactivity.suggestions.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proactivity.suggestions.SUGGESTIONS_DIR", tmp_path / "suggestions")
    monkeypatch.setattr(
        "proactivity.suggestions.CURRENT_PATH", tmp_path / "suggestions" / "current.md"
    )
    monkeypatch.setattr("proactivity.suggestions.ARCHIVE_DIR", tmp_path / "suggestions" / "archive")
    monkeypatch.setattr("proactivity.suggestions._LEGACY_PATH", tmp_path / "suggestions.md")
    monkeypatch.setattr("proactivity.promptlog.REPO_ROOT", tmp_path)
    monkeypatch.setattr("proactivity.promptlog.LOG_PATH", tmp_path / "proactivity_prompts.jsonl")
    monkeypatch.setattr("proactivity.promptlog._seq", 0)
    return tmp_path
