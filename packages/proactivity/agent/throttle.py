"""Throttle gates for the proactivity worker."""

from __future__ import annotations

import threading
import time


class AgentThrottle:
    """Burst / Claude / speak / memory timing gates (thread-safe)."""

    def __init__(
        self,
        *,
        min_consider_interval_s: float = 1.0,
        min_claude_interval_s: float = 0.0,
        min_speak_interval_s: float = 2.0,
        repeat_text_window_s: float = 30.0,
        min_memory_interval_s: float = 0.0,
    ) -> None:
        self.min_consider_interval_s = min_consider_interval_s
        self.min_claude_interval_s = min_claude_interval_s
        self.min_speak_interval_s = min_speak_interval_s
        self.repeat_text_window_s = repeat_text_window_s
        self.min_memory_interval_s = min_memory_interval_s

        self._lock = threading.Lock()
        self._last_classify_at: float = 0.0
        self._last_claude_at: float = 0.0
        self._last_speak_at: float = 0.0
        self._last_spoken_text: str | None = None
        self._last_spoken_at: float = 0.0
        self._last_memory_at: float = 0.0

    def try_consider(self) -> bool:
        """Record a classify attempt if burst throttle allows."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_classify_at < self.min_consider_interval_s:
                return False
            self._last_classify_at = now
            return True

    def try_claude(self) -> tuple[bool, str | None, float | None]:
        """Acquire a Claude slot if cooldown allows.

        Returns (ok, last_spoken_text, seconds_since_speak).
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_claude_at < self.min_claude_interval_s:
                return False, None, None
            self._last_claude_at = now
            last_spoken_text = self._last_spoken_text
            seconds_since_speak = now - self._last_spoken_at if self._last_spoken_at else None
            return True, last_spoken_text, seconds_since_speak

    def speak_gate_skip(self, text: str) -> str | None:
        """Return a skip reason if speak gates block this text."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_speak_at < self.min_speak_interval_s:
                return "speak_throttle"
            if (
                self._last_spoken_text == text
                and now - self._last_spoken_at < self.repeat_text_window_s
            ):
                return "repeat_text"
        return None

    def record_spoke(self, text: str) -> None:
        """Mark a successful speak for throttle bookkeeping."""
        now = time.monotonic()
        with self._lock:
            self._last_spoken_text = text
            self._last_spoken_at = now
            self._last_speak_at = now

    def try_memory(self) -> bool:
        """Acquire a memory snapshot slot if cooldown allows."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_memory_at < self.min_memory_interval_s:
                return False
            self._last_memory_at = now
            return True
