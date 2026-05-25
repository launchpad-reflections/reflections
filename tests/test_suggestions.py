"""Tests for proactivity.suggestions."""

from __future__ import annotations

from pathlib import Path

from proactivity import suggestions


def test_reset_and_append_suggestion(tmp_repo: Path) -> None:
    path = tmp_repo / "suggestions" / "current.md"
    suggestions.reset_suggestions(path)
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# Session suggestions")

    suggestions.append_suggestion("Try ramen nearby", reason="hungry", path=path)
    text = suggestions.read_suggestions(path)
    assert "Try ramen nearby" in text
    assert "reason: hungry" in text


def test_read_suggestions_empty_when_missing(tmp_repo: Path) -> None:
    path = tmp_repo / "suggestions" / "missing.md"
    assert suggestions.read_suggestions(path) == ""


def test_reset_archives_prior_session(tmp_repo: Path) -> None:
    path = tmp_repo / "suggestions" / "current.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# Session suggestions\n\n- [12:00:00] "hello"\n', encoding="utf-8")

    suggestions.reset_suggestions(path)
    archive_files = list((tmp_repo / "suggestions" / "archive").glob("*.md"))
    assert len(archive_files) == 1
    assert path.read_text(encoding="utf-8") == suggestions._HEADER
