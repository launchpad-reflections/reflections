"""Tests for proactivity.parser."""

from __future__ import annotations

from pathlib import Path

from proactivity.parser import parse_memory_file, parse_memory_md


def test_parse_entities_format() -> None:
    text = """\
## Entities

### Alex (self)
- aliases: Alex S
- prefers oat milk

### Sam (friend)
- severe nut allergy
"""
    entities, summaries = parse_memory_md(text)
    assert len(entities) == 2
    assert entities[0]["name"] == "Alex"
    assert entities[0]["relationship"] == "self"
    assert "prefers oat milk" in entities[0]["facts"]
    assert "Alex S" in entities[0]["aliases"]
    assert entities[1]["name"] == "Sam"
    assert summaries == []


def test_parse_legacy_people_section() -> None:
    text = """\
## People

- **Sam** (friend): likes ramen; lives nearby

## Context

Met at the cafe yesterday.
"""
    entities, summaries = parse_memory_md(text)
    assert len(entities) == 1
    assert entities[0]["name"] == "Sam"
    assert "likes ramen" in entities[0]["facts"]
    assert len(summaries) == 1
    assert summaries[0]["timestamp_approx"] == "context"


def test_parse_empty_returns_empty_lists() -> None:
    assert parse_memory_md("") == ([], [])
    assert parse_memory_md("   ") == ([], [])


def test_parse_memory_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    assert parse_memory_file(missing) == ([], [])


def test_parse_unrecognized_text_becomes_summary() -> None:
    text = "Just some free-form notes with no headers."
    entities, summaries = parse_memory_md(text)
    assert entities == []
    assert len(summaries) == 1
    assert summaries[0]["timestamp_approx"] == "session"
