"""Tests for proactivity.entities."""

from __future__ import annotations

from proactivity.entities import (
    extract_speakers,
    filter_entities_by_speakers,
    format_entities_compact,
)


def test_extract_speakers_user_first() -> None:
    transcript = [("Sam", "hi"), ("Alex", "hey"), ("Sam", "again")]
    speakers = extract_speakers(transcript, user_name="Alex")
    assert speakers == ["Alex", "Sam"]


def test_filter_entities_exact_and_alias_match() -> None:
    entities = [
        {"name": "Alex", "relationship": "self", "facts": ["wearer"], "aliases": []},
        {"name": "Samuel", "relationship": "friend", "facts": ["allergy"], "aliases": ["Sam"]},
    ]
    matched = filter_entities_by_speakers(entities, ["Sam"])
    assert len(matched) == 1
    assert matched[0]["name"] == "Samuel"


def test_filter_entities_fuzzy_one_char() -> None:
    entities = [
        {"name": "Alexis", "relationship": "friend", "facts": ["neighbor"], "aliases": []},
    ]
    matched = filter_entities_by_speakers(entities, ["Alexia"])
    assert len(matched) == 1


def test_format_entities_compact_empty() -> None:
    assert format_entities_compact([]) == "(no relevant entity info)"


def test_format_entities_compact_renders_facts() -> None:
    text = format_entities_compact(
        [
            {"name": "Sam", "relationship": "friend", "facts": ["nut allergy"]},
        ]
    )
    assert "Sam (friend): nut allergy" in text
