"""Entry point for the local stream viewer.

To add a pipeline: implement a Processor in stream/processors/ and
append it to PROCESSORS in main().
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

if TYPE_CHECKING:
    from proactivity.agent import ProactivityAgent

import cv2
from proactivity.memory_agent import MemoryAgent
from proactivity.speak import speak as _speak_phrase
from reflections.config import (
    PROACTIVITY_ENABLED,
    REPO_ROOT,
    SHOW_INTERIM,
    SONIOX_API_KEY,
    TRANSCRIPT_LOG_PATH,
    TRANSCRIPT_URL,
    USE_MENTRA_TRANSCRIPTION,
    USER_NAME,
    WHEP_URL,
)
from reflections.env import load_env
from reflections.logging_config import setup_logging
from stream.pipeline import Pipeline
from stream.processors.asd import ASDProcessor
from stream.processors.captions import CaptionProcessor
from stream.processors.soniox import SonioxProcessor
from stream.source import StreamSource

# `proactivity.classifier` is imported lazily inside `_build_session()`
# because constructing `ActionabilityClassifier` pulls in torch +
# transformers + peft. Users who run with `PROACTIVITY_ENABLED=0`
# (no classifier, no Claude) should not pay that import cost.

logger = logging.getLogger(__name__)


@dataclass
class ViewerSession:
    """Runtime state for one viewer session — created in main(), not at import."""

    user_name: str
    transcript_lines: list[tuple[str | None, str]] = field(default_factory=list)
    interim_last_len: int = 0
    transcript_event_count: int = 0
    prev_finalized_snapshot: list[tuple[str | None, str]] = field(default_factory=list)
    asd: ASDProcessor = field(default_factory=lambda: ASDProcessor(debug=False))
    captions: CaptionProcessor | None = None
    memory: MemoryAgent | None = None
    proactivity: ProactivityAgent | None = None
    processors: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.captions = CaptionProcessor(user_name=self.user_name)


def _print_interim(session: ViewerSession, text: str) -> None:
    padded = text.ljust(session.interim_last_len)
    sys.stdout.write(f"\r{padded}")
    sys.stdout.flush()
    session.interim_last_len = max(session.interim_last_len, len(text))


def _print_soniox(session: ViewerSession, text: str, speaker) -> None:
    session.transcript_lines.append((speaker, text))
    display = speaker if (speaker and speaker != "User") else session.user_name
    prefix = f"[{display}]"
    if session.interim_last_len:
        sys.stdout.write("\r" + " " * session.interim_last_len + "\r")
        session.interim_last_len = 0
    sys.stdout.write(f"{prefix} {text}\n")
    sys.stdout.flush()


def _on_transcript_update(session: ViewerSession, transcript: list[tuple[str | None, str]]) -> None:
    """Fires on speaker flip, sentence end, or 0.5s silence dwell."""
    assert session.captions is not None
    session.captions.update_transcript(transcript)
    if session.proactivity is not None:
        session.proactivity.consider(transcript)

    finalized = transcript[:-1] if len(transcript) > 1 else []
    if finalized == session.prev_finalized_snapshot:
        return
    session.prev_finalized_snapshot = list(finalized)
    session.transcript_event_count += 1
    lines = [
        f"[{spk if (spk and spk != 'User') else session.user_name}]: {text}"
        for spk, text in transcript
    ]
    with open(TRANSCRIPT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== Event #{session.transcript_event_count} ===\n")
        f.write("\n".join(lines))
        f.write("\n\n")


def _extract_name_mappings(
    session: ViewerSession,
    transcript: list[tuple[str | None, str]],
) -> dict[str, str]:
    """Call Claude to find real names for Person N speaker labels in the transcript."""
    person_labels = sorted({spk for spk, _ in transcript if spk and spk.startswith("Person ")})
    if not person_labels:
        return {}

    try:
        import anthropic
    except ImportError:
        logger.warning("[names] anthropic not installed; skipping name extraction")
        return {}

    lines = [
        f"[{spk if (spk and spk != 'User') else session.user_name}]: {text}"
        for spk, text in transcript
    ]
    transcript_str = "\n".join(lines)

    prompt = (
        "You are analyzing a conversation transcript to find real names for speakers "
        "labeled as 'Person N'.\n\n"
        f"Transcript:\n{transcript_str}\n\n"
        f"Speaker labels to resolve: {', '.join(person_labels)}\n\n"
        "Look for any moment where a speaker's real name is used — someone addresses "
        "them by name, they introduce themselves, or are referred to by name.\n\n"
        "Return a JSON object mapping only the labels you are confident about to real "
        f'names. Omit labels with no clear name. Example: {{"Person 1": "Bob"}}\n\n'
        "Return only the JSON object, nothing else."
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        mappings: dict = json.loads(raw)
        return {
            k: v.strip()
            for k, v in mappings.items()
            if k in person_labels and isinstance(v, str) and v.strip()
        }
    except Exception as e:
        logger.error("[names] extraction failed: %s", e)
        return {}


def _build_session() -> ViewerSession:
    session = ViewerSession(user_name=USER_NAME)
    session.processors = [session.asd, session.captions]

    session.memory = MemoryAgent(
        get_transcript=lambda: list(session.transcript_lines),
        memory_path=REPO_ROOT / "memory.md",
        user_name=USER_NAME,
    )

    if PROACTIVITY_ENABLED:
        from proactivity.agent import ProactivityAgent
        from proactivity.classifier import ActionabilityClassifier

        session.proactivity = ProactivityAgent(
            classifier=ActionabilityClassifier(),
            get_transcript=lambda: list(session.transcript_lines),
            memory_path=REPO_ROOT / "memory.md",
            user_name=USER_NAME,
            memory_agent=session.memory,
        )
    else:
        logger.info("[proactivity] disabled (PROACTIVITY_ENABLED=0)")

    if SONIOX_API_KEY:
        session.processors.append(
            SonioxProcessor(
                api_key=SONIOX_API_KEY,
                on_transcript=lambda text, speaker: _print_soniox(session, text, speaker),
                on_interim=lambda text: _print_interim(session, text),
                on_transcript_update=lambda transcript: _on_transcript_update(session, transcript),
                asd_processor=session.asd,
                user_name=USER_NAME,
            )
        )
    else:
        logger.warning(
            "[soniox] SONIOX_API_KEY not set; skipping Soniox processor",
        )

    return session


def _transcript_loop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            with urlopen(TRANSCRIPT_URL, timeout=10) as resp:
                logger.info("[transcripts] connected to %s", TRANSCRIPT_URL)
                for raw in resp:
                    if stop_event.is_set():
                        return
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue
                    is_final = bool(data.get("isFinal"))
                    if not is_final and not SHOW_INTERIM:
                        continue
                    tag = "FINAL" if is_final else "interim"
                    speaker = data.get("speakerId")
                    prefix = f"[{tag}]"
                    if speaker:
                        prefix += f"[{speaker}]"
                    print(f"{prefix} {text}", flush=True)
        except (URLError, ConnectionError, TimeoutError) as e:
            if stop_event.is_set():
                return
            logger.warning("[transcripts] disconnected (%s); retrying in 2s", e)
            stop_event.wait(2.0)
        except Exception as e:
            if stop_event.is_set():
                return
            logger.error("[transcripts] error: %s; retrying in 2s", e)
            stop_event.wait(2.0)


def main() -> None:
    load_env()
    setup_logging()

    parser = argparse.ArgumentParser(description="Reflections WHEP stream viewer")
    parser.add_argument(
        "--help-only",
        action="store_true",
        help="Print help and exit without connecting to the stream",
    )
    args = parser.parse_args()
    if args.help_only:
        parser.print_help()
        return

    session = _build_session()

    print(f"Connecting to {WHEP_URL} ...")
    source = StreamSource(WHEP_URL)
    pipeline = Pipeline(session.processors)
    pipeline.start()

    transcript_stop = threading.Event()
    transcript_thread = None
    if USE_MENTRA_TRANSCRIPTION:
        transcript_thread = threading.Thread(
            target=_transcript_loop, args=(transcript_stop,), name="transcripts", daemon=True
        )
        transcript_thread.start()

    try:
        for item in source.frames():
            if item.kind == "video":
                pipeline.dispatch_video(item.frame, item.pts)
                rendered = pipeline.draw(item.frame)
                cv2.imshow("Mentra Live", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and session.memory is not None:
                    threading.Thread(
                        target=session.memory.snapshot,
                        name="memory-snapshot",
                        daemon=True,
                    ).start()
                elif key in (ord("p"), ord("P")):
                    threading.Thread(
                        target=_speak_phrase,
                        name="glasses-speak",
                        daemon=True,
                    ).start()
                elif key in (ord("m"), ord("M")) and session.proactivity is not None:
                    new_state = not session.proactivity.is_enabled()
                    session.proactivity.set_enabled(new_state)
                    logger.info(
                        "[proactivity] %s",
                        "enabled" if new_state else "muted",
                    )
            else:
                pipeline.dispatch_audio(item.samples, item.sample_rate, item.pts)
    finally:
        transcript_stop.set()
        source.close()
        pipeline.stop()
        if session.proactivity is not None:
            session.proactivity.stop()
        if session.captions is not None:
            session.captions.clear()
        try:
            mappings = _extract_name_mappings(session, session.transcript_lines)
            if not mappings:
                logger.info("[names] no name mappings found")
            for person_label, real_name in mappings.items():
                if session.asd.rename_identity(person_label, real_name):
                    logger.info("[names] %s → %s", person_label, real_name)
                else:
                    logger.warning("[names] %s not in gallery (no rename)", person_label)
        except Exception as e:
            logger.error("[names] error: %s", e)
        try:
            session.asd.close()
        except Exception as e:
            logger.error("[main] asd close error: %s", e)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
