"""Tests for proactivity.render."""

from __future__ import annotations

from proactivity.render import (
    get_signal,
    render_entities,
    render_example,
    render_memory,
    render_tools,
    render_transcript,
)


def _sample_example() -> dict:
    return {
        "id": "t1",
        "label": 1,
        "reasoning": "User asked for nearby food.",
        "metadata": {"category": "location_dependent"},
        "entity_list": [
            {"name": "Alex", "relationship": "self", "facts": ["wears glasses"]},
        ],
        "transcript": {
            "turns": [
                {"speaker": "Sam", "text": "I'm hungry"},
                {"speaker": "Alex", "text": "want ramen?", "is_target": True},
            ],
        },
        "memory_summaries": [{"timestamp_approx": "today", "summary": "At campus."}],
        "available_tools": ["places_search", "send_message"],
        "location": {
            "description": "Indoor, Example City",
            "coordinates": {"latitude": 0.0, "longitude": 0.0},
            "nearby_places": [
                {"name": "Ramen House", "type": "restaurant", "distance_meters": 200},
            ],
        },
    }


def test_get_signal_maps_category() -> None:
    assert get_signal({"metadata": {"category": "location_dependent"}}) == "location"
    assert get_signal({"metadata": {"category": "neg_pleasantry"}}) == "transcript"
    assert get_signal({"metadata": {"category": "unknown_cat"}}) == "transcript"


def test_render_transcript_marks_target() -> None:
    text = render_transcript(_sample_example()["transcript"])
    assert "[TARGET] want ramen? [/TARGET]" in text
    assert "Sam: I'm hungry" in text


def test_render_entities_and_memory_empty() -> None:
    assert "(No known entity information.)" in render_entities([])
    assert "(No prior conversation context available.)" in render_memory([])


def test_render_tools_joins_names() -> None:
    assert render_tools(["a", "b"]) == "a, b"


def test_render_example_includes_chatml_tags() -> None:
    rendered = render_example(_sample_example())
    assert "<|im_start|>system" in rendered
    assert "<entities>" in rendered
    assert "<label>1</label>" in rendered
    assert "<signal>location</signal>" in rendered
