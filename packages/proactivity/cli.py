"""Interactive REPL for testing the proactivity pipeline without glasses.

Runs the EXACT same `ProactivityAgent` that `apps/viewer/cli.py` uses —
only the input source (stdin instead of Soniox) and the output sink
(stdout instead of glasses TTS) differ. Switching back to glasses is a
one-line change: stop running this driver and let `apps/viewer/cli.py`
construct the agent normally.

Usage:
    python -m proactivity.cli

REPL commands:
    /show     print the running transcript
    /clear    reset the transcript
    /mute     suppress speaks (classifier + Claude still run)
    /unmute   re-enable speaks
    /quit     exit

Plain lines append a turn to the transcript and trigger consider().
Optional `[Speaker Name] ` prefix sets a different speaker; default is
USER_NAME from .env.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from reflections.config import REPO_ROOT, USER_NAME
from reflections.env import load_env
from reflections.logging_config import setup_logging

from proactivity.agent import ProactivityAgent
from proactivity.classifier import ActionabilityClassifier
from proactivity.memory_agent import MemoryAgent

logger = logging.getLogger(__name__)

_DECISION_LOG = REPO_ROOT / "proactivity_decisions.log"


_SPEAKER_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


def _parse_line(line: str, default_speaker: str) -> tuple[str | None, str]:
    """`[Alice] hi there` → ('Alice', 'hi there'). Plain text uses the
    wearer's name. `default_speaker` is what the agent records for the
    wearer; passing the user_name keeps log lines symmetric with how
    `apps/viewer/cli.py` prints them."""
    m = _SPEAKER_RE.match(line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return default_speaker, line.strip()


def _tail_decision_log(stop_event: threading.Event, log_path: Path) -> None:
    """Pretty-print new decision-log lines so the user sees what the
    background worker decided after each consider() call. The agent
    writes to the log from a worker thread, so output appears slightly
    after the REPL prompt comes back — that's expected."""
    last_size = log_path.stat().st_size if log_path.exists() else 0
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
            if not log_path.exists():
                continue
            size = log_path.stat().st_size
            if size <= last_size:
                continue
            with open(log_path, encoding="utf-8") as f:
                f.seek(last_size)
                new_text = f.read()
            last_size = size
            for ln in new_text.splitlines():
                if ln.strip():
                    sys.stdout.write(f"\n  ↳ {ln}\n> ")
                    sys.stdout.flush()
    except Exception as e:
        logger.error("[cli] log tail error: %s", e)


def _print_help() -> None:
    logger.info(
        "\nCommands: /show /clear /mute /unmute /quit\n"
        "Default speaker is the wearer (USER_NAME). Prefix with "
        "[Name] to attribute a turn to someone else.\n"
        "Example: [Alice] who won the world cup in 2026\n",
    )


def main() -> int:
    load_env()
    setup_logging()
    user_name = USER_NAME
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "[cli] ANTHROPIC_API_KEY not set in .env — Claude calls " "will fail.",
        )

    logger.info("[cli] loading classifier (~3s)...")
    classifier = ActionabilityClassifier()

    transcript: list[tuple[str | None, str]] = []

    def on_propose(text: str, reason: str | None) -> None:
        # Fires for EVERY speak-true response, before throttle/mute
        # gates — so testing always shows what Claude proposed. The
        # ↳ log line later tells you whether a gate suppressed it.
        suffix = f"  ({reason})" if reason else ""
        sys.stdout.write(f"\n🔊 [GLASSES SAYS]: {text}{suffix}\n> ")
        sys.stdout.flush()

    def speak_fn(text: str) -> bool:
        # Quiet — on_propose already printed the line. Returning True
        # so the agent records spoke=ok in the decision log.
        return True

    memory = MemoryAgent(
        get_transcript=lambda: list(transcript),
        memory_path=REPO_ROOT / "memory.md",
        user_name=user_name,
    )

    agent = ProactivityAgent(
        classifier=classifier,
        get_transcript=lambda: list(transcript),
        memory_path=REPO_ROOT / "memory.md",
        user_name=user_name,
        speak_fn=speak_fn,
        on_propose=on_propose,
        log_path=_DECISION_LOG,
        memory_agent=memory,
    )

    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=_tail_decision_log,
        args=(stop_event, _DECISION_LOG),
        name="decision-log-tail",
        daemon=True,
    )
    tail_thread.start()

    logger.info(
        "\n[cli] ready. wearer = %r. type a turn or /help.",
        user_name,
    )
    _print_help()

    try:
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/help":
                _print_help()
                continue
            if line == "/show":
                if not transcript:
                    print("(empty)")
                for spk, text in transcript:
                    print(f"  [{spk or user_name}]: {text}")
                continue
            if line == "/clear":
                transcript.clear()
                print("(cleared)")
                continue
            if line == "/mute":
                agent.set_enabled(False)
                print("(muted — classifier still runs, speaks suppressed)")
                continue
            if line == "/unmute":
                agent.set_enabled(True)
                print("(unmuted)")
                continue

            speaker, text = _parse_line(line, user_name)
            if not text:
                continue
            transcript.append((speaker, text))
            agent.consider(transcript)
    finally:
        stop_event.set()
        agent.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
