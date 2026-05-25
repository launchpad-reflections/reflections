"""End-to-end probe of the proactivity pipeline.

Drives the same `ProactivityAgent` cli.py uses, but feeds it pre-canned
transcripts synchronously (bypassing the size-1 inbox) so we can read
back exactly what the classifier scored, what Claude was asked, what
tools fired, and what would have been spoken.

Run:
    python scripts/smoke_pipeline.py
"""

from __future__ import annotations

import os
import sys

from proactivity.agent import ProactivityAgent  # noqa: E402
from proactivity.classifier import ActionabilityClassifier  # noqa: E402
from reflections.config import REPO_ROOT
from reflections.env import load_env
from reflections.logging_config import setup_logging

# Each probe sets the running transcript context. The LAST line is the
# target sentence (the one the classifier scores). `expect` is what we
# *hope* happens — used only for the readout, not enforced.
PROBES: list[dict] = [
    {
        "name": "anticipate_dinner",
        "expect": "places_search without asking permission, short reply",
        "turns": [
            ("Sam", "I'm hungry"),
            ("Alex", "yeah, should we get dinner somewhere"),
        ],
    },
    {
        "name": "opinion_request",
        "expect": "speak with a confident pick (not silent)",
        "turns": [
            ("Sam", "should we go to a movie tonight or just stay in"),
        ],
    },
    {
        "name": "smalltalk_ignore",
        "expect": "should be silent",
        "turns": [
            ("Alex", "yo what's up"),
            ("Sam", "not much, just chilling"),
        ],
    },
    {
        "name": "general_knowledge_no_tool",
        "expect": "speak from general knowledge, no tool",
        "turns": [
            ("Alex", "wait what year did world war two end"),
        ],
    },
    {
        "name": "places_nearby_query",
        "expect": "places_search tool",
        "turns": [
            ("Sam", "I'm starving"),
            ("Alex", "what's a good ramen place nearby"),
        ],
    },
    {
        "name": "directions_query",
        "expect": "directions tool",
        "turns": [
            ("Alex", "how long would it take to walk to the city library from here"),
        ],
    },
    {
        "name": "current_event_websearch",
        "expect": "web_search tool",
        "turns": [
            ("Sam", "did the warriors play last night"),
            ("Alex", "yeah I think so but I don't know the score"),
        ],
    },
    {
        "name": "explicit_search_request",
        "expect": "web_search tool",
        "turns": [
            ("Alex", "please search what the transit fare from downtown to the airport is"),
        ],
    },
    {
        "name": "place_followup_question",
        "expect": "places_search tool",
        "turns": [
            ("Sam", "I want coffee"),
            ("Alex", "where's a coffee shop open right now near campus"),
        ],
    },
    {
        "name": "calendar_today_query",
        "expect": "list_calendar_events tool",
        "turns": [
            ("Alex", "what do I have on my calendar today"),
        ],
    },
    {
        "name": "calendar_find_event",
        "expect": "find_calendar_event tool",
        "turns": [
            ("Sam", "didn't you say you had a dentist appt soon"),
            ("Alex", "yeah I forget when it is"),
        ],
    },
    {
        "name": "calendar_create_event",
        "expect": "create_calendar_event tool",
        "turns": [
            ("Sam", "wanna grab dinner tomorrow at 7"),
            ("Alex", "yeah put dinner with sam on my calendar tomorrow at 7pm"),
        ],
    },
]


def _print_probe_header(i: int, probe: dict) -> None:
    print()
    print("=" * 78)
    print(f"PROBE {i + 1}/{len(PROBES)}: {probe['name']}")
    print(f"  expect: {probe['expect']}")
    for spk, text in probe["turns"]:
        print(f"  [{spk}]: {text}")
    print("-" * 78)


def main() -> int:
    load_env()
    setup_logging()
    user_name = os.environ.get("USER_NAME") or "User"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[smoke] ANTHROPIC_API_KEY missing — Claude calls will fail.")
        return 2
    if not os.environ.get("GOOGLE_MAPS_API_KEY"):
        print("[smoke] WARN: GOOGLE_MAPS_API_KEY missing — maps tools will error.")
    if not all(
        os.environ.get(k)
        for k in (
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "GOOGLE_OAUTH_REFRESH_TOKEN",
        )
    ):
        print("[smoke] WARN: GOOGLE_OAUTH_* missing — calendar tools will error.")

    decision_log = REPO_ROOT / "proactivity_test_decisions.log"
    decision_log.unlink(missing_ok=True)

    print("[smoke] loading classifier (~3-30s on first run)...")
    classifier = ActionabilityClassifier()

    transcript: list[tuple[str | None, str]] = []

    proposed: list[tuple[str, str | None]] = []

    def on_propose(text: str, reason: str | None) -> None:
        proposed.append((text, reason))

    def speak_fn(_text: str) -> bool:
        return True

    agent = ProactivityAgent(
        classifier=classifier,
        get_transcript=lambda: list(transcript),
        memory_path=REPO_ROOT / "memory.md",
        user_name=user_name,
        speak_fn=speak_fn,
        on_propose=on_propose,
        log_path=decision_log,
        # Reset throttles so each probe is independent.
        min_consider_interval_s=0.0,
        min_claude_interval_s=0.0,
        min_speak_interval_s=0.0,
        repeat_text_window_s=0.0,
    )

    summary: list[dict] = []

    for i, probe in enumerate(PROBES):
        _print_probe_header(i, probe)

        # Reset state for clean per-probe accounting.
        transcript.clear()
        proposed.clear()
        log_size_before = decision_log.stat().st_size if decision_log.exists() else 0

        for spk, text in probe["turns"]:
            transcript.append((spk, text))

        # Drive the worker synchronously to keep tests deterministic.
        agent._run_pipeline(list(transcript))  # noqa: SLF001

        # Drain the decision log line(s) appended this probe.
        new_lines: list[str] = []
        if decision_log.exists():
            with open(decision_log, encoding="utf-8") as f:
                f.seek(log_size_before)
                new_lines = [ln for ln in f.read().splitlines() if ln.strip()]

        for ln in new_lines:
            print(f"  log: {ln}")
        for text, reason in proposed:
            suffix = f"  ({reason})" if reason else ""
            print(f"  GLASSES → {text}{suffix}")

        summary.append(
            {
                "name": probe["name"],
                "expect": probe["expect"],
                "log": new_lines,
                "proposed": list(proposed),
            }
        )

    agent.stop()

    # ---- Summary table.
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for row in summary:
        log_text = " | ".join(row["log"]) if row["log"] else "(no log)"
        print(f"\n• {row['name']}  ({row['expect']})")
        print(f"    log: {log_text}")
        for text, reason in row["proposed"]:
            print(f"    spoke → {text}  ({reason or '-'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
