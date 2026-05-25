"""Two-stage proactivity agent for the smart-glasses viewer.

Pipeline (per finalized transcript update):
  1. Soniox calls _on_transcript_update on its asyncio thread.
  2. apps/viewer/cli.py calls ProactivityAgent.consider(transcript) —
     pushes onto a size-1 queue and returns immediately (must be < 1 ms;
     the asyncio thread cannot block).
  3. A daemon worker thread pops the latest snapshot, runs the local
     Qwen 3 1.7B + LoRA gate (~200 ms).
  4. If the classifier says actionable (P >= threshold), build a
     prompt and ask Claude Haiku 4.5 whether to speak and what to say.
  5. If Claude returns speak=true, POST /speak on the Bun app server,
     which forwards to the glasses via session.audio.speak().

All decisions (drop / claude / spoke) append a single line to
proactivity_decisions.log so the prompt + thresholds can be tuned
without instrumenting from scratch.

Threading:
  - One internal daemon thread runs the worker loop forever.
  - A size-1 queue means rapid-fire considers naturally evict stale
    transcripts — only the freshest snapshot ever gets processed.
  - Throttle/state fields are guarded by a single Lock.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reflections.config import GLASSES_GATE_THRESHOLD

from proactivity.entities import (
    extract_speakers,
    filter_entities_by_speakers,
    format_entities_compact,
)
from proactivity.parser import parse_memory_file
from proactivity.promptlog import log_event
from proactivity.speak import speak as default_speak
from proactivity.suggestions import (
    append_suggestion,
    read_suggestions,
    reset_suggestions,
)

if TYPE_CHECKING:
    # Heavy import (torch/transformers/peft) — kept out of module load so
    # callers that don't enable proactivity never pay the price.
    from proactivity.classifier import ActionabilityClassifier

from .claude_loop import call_claude
from .prompts import DEFAULT_MODEL, DEFAULT_TOOLS
from .throttle import AgentThrottle

logger = logging.getLogger(__name__)


@dataclass
class _Decision:
    """One row in the decision log."""

    classify_p: float
    classify_label: int
    speak: bool
    spoke_ok: bool | None
    spoke_text: str | None
    classify_ms: float
    claude_ms: float | None
    speak_ms: float | None
    skip_reason: str | None
    claude_reason: str | None
    tool_calls: int = 0
    tool_names: list[str] | None = None


class ProactivityAgent:
    def __init__(
        self,
        *,
        classifier: ActionabilityClassifier,
        get_transcript: Callable[[], list[tuple[str | None, str]]],
        memory_path: str | Path = "memory.md",
        user_name: str = "User",
        speak_fn: Callable[[str], bool] = default_speak,
        on_propose: Callable[[str, str | None], None] | None = None,
        log_path: str | Path = "proactivity_decisions.log",
        anthropic_model: str = DEFAULT_MODEL,
        threshold: float = GLASSES_GATE_THRESHOLD,
        min_consider_interval_s: float = 1.0,
        min_claude_interval_s: float = 0.0,
        min_speak_interval_s: float = 2.0,
        repeat_text_window_s: float = 30.0,
        recent_turns: int = 10,
        recent_seconds: float = 30.0,
        tools: list[str] | None = None,
        memory_agent: Any | None = None,
        min_memory_interval_s: float = 0.0,
    ):
        self.classifier = classifier
        self.get_transcript = get_transcript
        self.memory_path = Path(memory_path)
        self.user_name = user_name
        self.speak_fn = speak_fn
        self.on_propose = on_propose
        self.log_path = Path(log_path)
        self.anthropic_model = anthropic_model
        self.threshold = threshold
        self.recent_turns = recent_turns
        self.recent_seconds = recent_seconds
        self.tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
        self.memory_agent = memory_agent

        self._throttle = AgentThrottle(
            min_consider_interval_s=min_consider_interval_s,
            min_claude_interval_s=min_claude_interval_s,
            min_speak_interval_s=min_speak_interval_s,
            repeat_text_window_s=repeat_text_window_s,
            min_memory_interval_s=min_memory_interval_s,
        )

        self._state_lock = threading.Lock()
        self._enabled = True

        reset_suggestions()

        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="proactivity",
            daemon=True,
        )
        self._worker.start()

        logger.info("[proactivity] ready")

    # -------------------------------------------------------------- public

    def consider(self, transcript: list[tuple[str | None, str]]) -> None:
        """Public entry-point — called by Soniox's asyncio thread."""
        if not transcript:
            return
        snapshot = list(transcript)
        try:
            self._inbox.put_nowait(snapshot)
        except queue.Full:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                pass
            try:
                self._inbox.put_nowait(snapshot)
            except queue.Full:
                pass

    def set_enabled(self, enabled: bool) -> None:
        with self._state_lock:
            self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._state_lock:
            return self._enabled

    def stop(self) -> None:
        self._stop.set()
        try:
            self._inbox.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------- worker

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                continue
            try:
                self._run_pipeline(item)
            except Exception as e:
                logger.error("[proactivity] worker error: %s", e)

    def _run_pipeline(self, transcript: list[tuple[str | None, str]]) -> None:
        if not self._throttle.try_consider():
            return

        recent = self._select_recent(transcript)
        if not recent:
            return
        full_entity_list, memory_summaries = parse_memory_file(self.memory_path)

        speakers = extract_speakers(recent, user_name=self.user_name)
        entity_list = filter_entities_by_speakers(full_entity_list, speakers)

        turns_for_classifier = [
            {"speaker": (spk if spk else self.user_name), "text": text} for spk, text in recent
        ]

        t0 = time.monotonic()
        try:
            p, label, _ = self.classifier.classify(
                turns=turns_for_classifier,
                memory_summaries=memory_summaries,
                entity_list=entity_list,
                tools=self.tools,
                label_only=True,
            )
        except Exception as e:
            logger.error("[proactivity] classify failed: %s", e)
            return
        classify_ms = (time.monotonic() - t0) * 1000.0

        passed = p >= self.threshold
        log_event(
            "classifier",
            "gate",
            {
                "p": p,
                "threshold": self.threshold,
                "passed": passed,
                "classify_ms": classify_ms,
            },
        )

        if not passed:
            self._log(
                _Decision(
                    classify_p=p,
                    classify_label=label,
                    speak=False,
                    spoke_ok=None,
                    spoke_text=None,
                    classify_ms=classify_ms,
                    claude_ms=None,
                    speak_ms=None,
                    skip_reason="below_threshold",
                    claude_reason=None,
                )
            )
            return

        ok, last_spoken_text, seconds_since_speak = self._throttle.try_claude()
        if not ok:
            self._log(
                _Decision(
                    classify_p=p,
                    classify_label=label,
                    speak=False,
                    spoke_ok=None,
                    spoke_text=None,
                    classify_ms=classify_ms,
                    claude_ms=None,
                    speak_ms=None,
                    skip_reason="claude_throttle",
                    claude_reason=None,
                )
            )
            return

        with self._state_lock:
            enabled = self._enabled

        entities_block = format_entities_compact(entity_list)
        t0 = time.monotonic()
        decision = call_claude(
            anthropic_model=self.anthropic_model,
            user_name=self.user_name,
            entities_block=entities_block,
            recent_transcript=self._format_recent(recent),
            classifier_p=p,
            suggestions=read_suggestions(),
            last_spoken_text=last_spoken_text,
            seconds_since_speak=seconds_since_speak,
        )
        claude_ms = (time.monotonic() - t0) * 1000.0

        self._maybe_update_memory()

        speak = bool(decision.get("speak"))
        text = (decision.get("text") or "").strip() if speak else ""
        claude_reason = (decision.get("reason") or "").strip() or None
        tool_calls = int(decision.get("_tool_calls", 0))
        tool_names = list(decision.get("_tool_names", []) or []) or None

        if not speak or not text:
            self._log(
                _Decision(
                    classify_p=p,
                    classify_label=label,
                    speak=False,
                    spoke_ok=None,
                    spoke_text=None,
                    classify_ms=classify_ms,
                    claude_ms=claude_ms,
                    speak_ms=None,
                    skip_reason="claude_silent",
                    claude_reason=claude_reason,
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                )
            )
            return

        append_suggestion(text, reason=claude_reason, tool_names=tool_names)

        if self.on_propose is not None:
            try:
                self.on_propose(text, claude_reason)
            except Exception as e:
                logger.error("[proactivity] on_propose raised: %s", e)

        skip = self._throttle.speak_gate_skip(text)
        if skip:
            self._log(
                _Decision(
                    classify_p=p,
                    classify_label=label,
                    speak=True,
                    spoke_ok=False,
                    spoke_text=text,
                    classify_ms=classify_ms,
                    claude_ms=claude_ms,
                    speak_ms=None,
                    skip_reason=skip,
                    claude_reason=claude_reason,
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                )
            )
            return

        if not enabled:
            self._log(
                _Decision(
                    classify_p=p,
                    classify_label=label,
                    speak=True,
                    spoke_ok=False,
                    spoke_text=text,
                    classify_ms=classify_ms,
                    claude_ms=claude_ms,
                    speak_ms=None,
                    skip_reason="muted",
                    claude_reason=claude_reason,
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                )
            )
            return

        t0 = time.monotonic()
        ok = False
        try:
            ok = bool(self.speak_fn(text))
        except Exception as e:
            logger.error("[proactivity] speak failed: %s", e)
        speak_ms = (time.monotonic() - t0) * 1000.0

        if ok:
            self._throttle.record_spoke(text)

        self._log(
            _Decision(
                classify_p=p,
                classify_label=label,
                speak=True,
                spoke_ok=ok,
                spoke_text=text,
                classify_ms=classify_ms,
                claude_ms=claude_ms,
                speak_ms=speak_ms,
                skip_reason=None,
                claude_reason=claude_reason,
                tool_calls=tool_calls,
                tool_names=tool_names,
            )
        )

    # ------------------------------------------------------- helpers

    def _maybe_update_memory(self) -> None:
        if self.memory_agent is None:
            return
        if not self._throttle.try_memory():
            return

        threading.Thread(
            target=self.memory_agent.snapshot,
            name="memory-snapshot",
            daemon=True,
        ).start()

    def _select_recent(
        self,
        transcript: list[tuple[str | None, str]],
    ) -> list[tuple[str | None, str]]:
        if not transcript:
            return []
        return list(transcript[-self.recent_turns :])

    def _format_recent(
        self,
        recent: list[tuple[str | None, str]],
    ) -> str:
        return "\n".join(f"[{spk if spk else self.user_name}]: {text}" for spk, text in recent)

    # ---------------------------------------------------------- logging

    def _log(self, d: _Decision) -> None:
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
        parts = [
            ts,
            f"p={d.classify_p:.3f}",
            f"label={d.classify_label}",
            f"classify_ms={d.classify_ms:.0f}",
        ]
        if d.claude_ms is not None:
            parts.append(f"claude_ms={d.claude_ms:.0f}")
        if d.speak_ms is not None:
            parts.append(f"speak_ms={d.speak_ms:.0f}")
        if d.tool_calls:
            parts.append(f"tool_calls={d.tool_calls}")
            if d.tool_names:
                parts.append(f"tools={','.join(d.tool_names)}")
        if d.skip_reason:
            parts.append(f"skip={d.skip_reason}")
        if d.claude_reason:
            parts.append(f'reason="{d.claude_reason}"')
        if d.spoke_text:
            parts.append(f'text="{d.spoke_text}"')
        if d.spoke_ok is True:
            parts.append("spoke=ok")
        elif d.spoke_ok is False and not d.skip_reason:
            parts.append("spoke=fail")
        line = " | ".join(parts) + "\n"
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.error("[proactivity] log write failed: %s", e)
