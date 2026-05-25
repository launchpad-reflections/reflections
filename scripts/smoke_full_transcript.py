"""Run the actionability classifier on a full transcript with memory.md
as the memory context. Edit TRANSCRIPT, ENTITY_LIST, and TOOLS at the
top of this file to probe new scenarios.

Usage:
    python scripts/smoke_full_transcript.py
"""

from __future__ import annotations

from proactivity.classifier import classify, load_model  # noqa: E402
from reflections.config import REPO_ROOT
from reflections.env import load_env
from reflections.logging_config import setup_logging

# Speaker | text. Last line is the target (the sentence being classified).
# Fully synthetic dialogue — edit freely to probe new scenarios.
TRANSCRIPT = """\
Alex | hey, glad we could catch up
Sam | yeah it's been a while, how's the project going
Alex | good, the streaming pipeline is mostly working
Sam | nice, are we still on for dinner tomorrow
Alex | yeah I was thinking Thai
Sam | sounds good, but no peanuts for me
Alex | right, totally forgot — let me find a place that does nut-free
Sam | appreciate it
Alex | what's a good thai restaurant nearby that can do nut-free
"""


# Entity list — derived from memory.md plus reasonable defaults for the
# wearer and the friends present. All names are fictional placeholders.
ENTITY_LIST = [
    {
        "name": "Alex",
        "relationship": "self",
        "facts": ["wears the smart glasses", "building reflections app"],
    },
    {
        "name": "Sam",
        "relationship": "friend",
        "facts": ["severe nut allergy", "prefers Thai food without peanuts"],
    },
]


# Tools available to act on this turn.
TOOLS = ["send_message"]


# Approximate location (matches classifier default; override via env vars).
LOCATION = {
    "description": "Indoor, Example City CA",
    "coordinates": {"latitude": 0.0, "longitude": 0.0},
    "nearby_places": [],
}


def parse_transcript(s: str) -> list[dict]:
    """Convert pipe-separated 'speaker | text' lines into turn dicts."""
    turns = []
    for line in s.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        speaker, _, text = line.partition("|")
        turns.append({"speaker": speaker.strip(), "text": text.strip()})
    if turns:
        turns[-1]["is_target"] = True
    return turns


def main() -> None:
    load_env()
    setup_logging()
    memory_path = REPO_ROOT / "memory.md"
    memory_content = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""

    turns = parse_transcript(TRANSCRIPT)
    target_idx = len(turns) - 1
    target_text = turns[target_idx]["text"]

    example = {
        "id": "full_transcript_test",
        "transcript": {
            "turns": turns,
            "target_speaker": turns[target_idx]["speaker"],
            "target_index": target_idx,
        },
        "memory_summaries": (
            [{"timestamp_approx": "current_session", "summary": memory_content.strip()}]
            if memory_content.strip()
            else []
        ),
        "available_tools": TOOLS,
        "location": LOCATION,
        "entity_list": ENTITY_LIST,
        # Required by render_example() but unused at inference time.
        "label": 0,
        "reasoning": "(test)",
        "metadata": {
            "category": "memory_dependent",
            "subcategory": "test",
            "difficulty": "medium",
            "signals_used": ["memory", "transcript", "entities"],
            "action_type": None,
        },
    }

    print(
        f"[setup] turns: {len(turns)}, memory chars: {len(memory_content)}, "
        f"entities: {len(ENTITY_LIST)}, tools: {TOOLS}"
    )
    print(f"[setup] target: {target_text!r}\n")

    model, tokenizer, device, t0_id, t1_id = load_model()
    p, label, reasoning = classify(model, tokenizer, device, t0_id, t1_id, example)

    print("=== RESULT ===")
    bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
    print(f"P(actionable) = {p:.4f}  [{bar}]  threshold=0.45")
    print(f"Decision: {'🟢 ACTIONABLE → pass to LLM' if label == 1 else '🔴 NOT ACTIONABLE'}")
    if reasoning:
        print(f"Reasoning: {reasoning.strip()}")


if __name__ == "__main__":
    main()
