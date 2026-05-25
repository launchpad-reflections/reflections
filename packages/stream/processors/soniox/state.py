"""Transcript state: interim/final tracking and update emissions."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

# Characters that end a sentence.
_SENTENCE_TERMINATORS = ".?!"

# Debounce: new ASD-reported speaker must persist this long before we
# accept a flip, so brief misattributions don't churn the transcript.
_SPEAKER_FLIP_DEBOUNCE_S = 0.2

# Silence dwell: if the in-progress sentence hasn't grown for this long,
# treat it as a sentence boundary and fire an update.
_DWELL_S = 0.5

# Floor on time between emissions of the same kind — prevents token-rate
# chatter from flooding downstream handlers.
_MIN_EMIT_GAP_S = 0.25

logger = logging.getLogger(__name__)


def _find_terminator(s: str) -> int | None:
    """Index of the earliest sentence terminator in s, or None."""
    best = None
    for ch in _SENTENCE_TERMINATORS:
        i = s.find(ch)
        if i != -1 and (best is None or i < best):
            best = i
    return best


class TranscriptStateMixin:
    """Live transcript state machine and on_transcript_update emissions."""

    asd: Any
    debug: bool
    on_transcript_update: Callable[[list[tuple[str | None, str]]], None] | None
    _stopped: threading.Event

    def _init_transcript_state(self) -> None:
        # Transcript-update state. All read/written under _tx_lock because
        # the dwell watchdog (asyncio task) and _recv_loop both touch it,
        # and the user-supplied callback may inspect state synchronously.
        self._tx_lock = threading.Lock()
        self._finalized: list[tuple[str | None, str]] = []
        self._sentence_finals: str = ""  # finalized text for current sentence
        self._interim_text: str = ""  # latest non-final tail
        self._live_speaker: str | None = None
        self._pending_speaker: str | None = None
        self._pending_speaker_since: float = 0.0
        self._live_last_growth_mono: float = time.monotonic()
        self._last_emit_signature: tuple | None = None
        self._last_emit_mono: float = 0.0

    def _ingest_tokens(
        self,
        *,
        new_final_text: str,
        interim: str,
        final_speaker: str | None,
        endpoint_hit: bool,
    ) -> None:
        """Update live-transcript state from a Soniox message. Fires
        on_transcript_update when a trigger condition is met."""
        now = time.monotonic()
        speaker_flipped = False
        sentence_committed = False

        # Live-speaker guess: prefer the authoritative ASD window lookup
        # when we have fresh finals; fall back to "who's speaking right
        # now?" for interim-only updates.
        if final_speaker is not None:
            candidate = final_speaker
        elif self.asd is not None and interim:
            try:
                candidate = self.asd.who_is_speaking_now(window_s=0.3) or self._live_speaker
            except Exception as e:
                logger.warning("[soniox] live-speaker lookup failed: %s", e)
                candidate = self._live_speaker
        else:
            candidate = self._live_speaker

        with self._tx_lock:
            # Speaker debounce: a candidate must persist ≥200 ms before
            # we accept it as a flip. Finalized-segment speakers bypass
            # the debounce (they're already resolved over a window).
            accept_flip = False
            if candidate is not None and candidate != self._live_speaker:
                if final_speaker is not None:
                    accept_flip = True
                elif candidate == self._pending_speaker:
                    if now - self._pending_speaker_since >= _SPEAKER_FLIP_DEBOUNCE_S:
                        accept_flip = True
                else:
                    self._pending_speaker = candidate
                    self._pending_speaker_since = now
            else:
                self._pending_speaker = None

            if accept_flip:
                live_text = (self._sentence_finals + self._interim_text).strip()
                # Only commit on a speaker flip if the text ends at a real sentence
                # boundary. Partial utterances from brief ASD misattributions would
                # otherwise create permanent fragments in _finalized — the complete
                # sentence will arrive from Soniox shortly and commit correctly.
                if live_text and live_text[-1] in _SENTENCE_TERMINATORS:
                    self._finalized.append((self._live_speaker, live_text))
                self._sentence_finals = ""
                self._interim_text = ""
                self._live_speaker = candidate
                self._pending_speaker = None
                speaker_flipped = True

            # Append newly-finalized text to current sentence.
            if new_final_text:
                self._sentence_finals += new_final_text
                self._live_last_growth_mono = now

            # Update interim (only bump growth clock when content changed).
            if interim != self._interim_text:
                self._interim_text = interim
                if interim:
                    self._live_last_growth_mono = now

            # Commit any sentence-terminated prefixes out of the
            # finalized portion. Interim-only terminators still fire the
            # event (below) but don't commit — they'll commit once
            # finalized.
            while True:
                idx = _find_terminator(self._sentence_finals)
                if idx is None:
                    break
                sentence = self._sentence_finals[: idx + 1].strip()
                remainder = self._sentence_finals[idx + 1 :].lstrip()
                if sentence:
                    self._finalized.append((self._live_speaker, sentence))
                    sentence_committed = True
                self._sentence_finals = remainder

            interim_terminator = _find_terminator(self._interim_text) is not None

        # Fire the update event if any trigger condition applies. Emit
        # dedup (signature + min-gap) prevents floods when multiple
        # triggers fire in the same message.
        if speaker_flipped or sentence_committed or interim_terminator or endpoint_hit:
            self._emit_update()

    def _flush_live(self, *, reason: str) -> None:
        """Move any in-progress sentence into finalized history. Called
        on shutdown so the trailing utterance isn't dropped."""
        with self._tx_lock:
            live = (self._sentence_finals + self._interim_text).strip()
            if live:
                self._finalized.append((self._live_speaker, live))
            self._sentence_finals = ""
            self._interim_text = ""
            if self.debug:
                logger.debug("[soniox] flushed live (%s)", reason)

    async def _dwell_loop(self) -> None:
        """Fire an update when the in-progress sentence hasn't grown for
        _DWELL_S — interpret sustained quiet as a sentence boundary."""
        while not self._stopped.is_set():
            await asyncio.sleep(0.1)
            with self._tx_lock:
                has_live = bool(self._sentence_finals or self._interim_text)
                quiet = time.monotonic() - self._live_last_growth_mono >= _DWELL_S
            if has_live and quiet:
                self._emit_update()

    def _current_snapshot(self) -> list[tuple[str | None, str]]:
        """Full cumulative transcript including the in-progress sentence
        (caller must hold _tx_lock)."""
        out = list(self._finalized)
        live = (self._sentence_finals + self._interim_text).strip()
        if live:
            out.append((self._live_speaker, live))
        return out

    def _emit_update(self, *, force: bool = False) -> None:
        if self.on_transcript_update is None:
            return
        with self._tx_lock:
            snapshot = self._current_snapshot()
            signature = tuple(snapshot)
            now = time.monotonic()
            if signature == self._last_emit_signature and not force:
                return
            if not force and now - self._last_emit_mono < _MIN_EMIT_GAP_S:
                return
            self._last_emit_signature = signature
            self._last_emit_mono = now
        try:
            self.on_transcript_update(list(snapshot))
        except Exception as e:
            logger.error("[soniox] on_transcript_update raised: %s", e)

    def get_transcript(self) -> list[tuple[str | None, str]]:
        """Thread-safe snapshot of the full cumulative transcript,
        including any in-progress sentence."""
        with self._tx_lock:
            return self._current_snapshot()
