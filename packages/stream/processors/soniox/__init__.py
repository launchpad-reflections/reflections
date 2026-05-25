"""Persistent Soniox real-time transcription processor.

Bypasses MentraOS's VAD-gated transcription (which tears down the
Soniox WebSocket on silence and pays a ~1s cold start on reopen) by
holding a single WebSocket open for the life of the session and
streaming the glasses mic audio directly.

Audio from aiortc arrives as 48 kHz signed 16-bit samples (stored in
float32 at int16 scale). We downsample 48k -> 16k (exact 3:1) and
send raw PCM s16le frames.

Speaker attribution: each Soniox token carries a per-token speaker id
(audio-side diarization). We fuse that with face-side ASD identities
via a per-spk_id score table — see _bind_soniox_speaker_to_identity.
Until the binding locks, segments are labeled with ASD's running best
guess; once locked, the spk_id always renders as that identity. When
ASD reports no speaking face during a segment, that's evidence the
glasses-wearer is the source, so the wearer label gains weight.
Unresolved face tracks never leak into the transcript as 'Track N' —
they fall through to the wearer label and let the fusion table catch
up over the next few segments.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from stream.pipeline import Processor

from .fusion import SpeakerFusionMixin
from .state import TranscriptStateMixin
from .ws import SonioxWebSocketMixin

logger = logging.getLogger(__name__)

__all__ = ["SonioxProcessor"]


class SonioxProcessor(
    Processor,
    SonioxWebSocketMixin,
    SpeakerFusionMixin,
    TranscriptStateMixin,
):
    mode = "async"
    name = "soniox"
    consumes = frozenset({"audio"})

    def __init__(
        self,
        api_key: str,
        on_transcript: Callable[[str, str | None], None] | None = None,
        on_interim: Callable[[str], None] | None = None,
        on_transcript_update: Callable[[list[tuple[str | None, str]]], None] | None = None,
        model: str = "stt-rt-v4",
        asd_processor: Any | None = None,
        user_name: str = "User",
        debug: bool = False,
    ):
        self.api_key = api_key
        self.on_transcript = on_transcript
        self.on_interim = on_interim
        self.on_transcript_update = on_transcript_update
        self.model = model
        self.debug = debug
        self.asd = asd_processor
        # Wearer label. Used both as the default speaker (when there's
        # no diarization signal) and as the candidate that accumulates
        # W_SILENT_FACE_USER weight whenever Soniox reports speech but
        # ASD sees no speaking face.
        self.user_name = user_name

        self._init_fusion_state()
        self._init_transcript_state()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_q: asyncio.Queue | None = None
        self._loop_ready = threading.Event()
        self._stopped = threading.Event()
        self._last_final_end_ms = 0
        self._drops = 0

        self._stream_ms_sent: float = 0.0
        self._clock_lock = threading.Lock()
        # Anchor for converting Soniox stream_ms → monotonic (wall-clock
        # seconds). Updated on every audio ingestion: pair of
        # (stream_ms_after_ingest, monotonic_at_ingest). Since audio
        # flows in real time, mono(ms) ≈ anchor_mono - (anchor_ms - ms)/1000.
        self._clock_anchor: tuple[float, float] | None = None

        self._thread = threading.Thread(target=self._thread_main, name="soniox-loop", daemon=True)
        self._thread.start()
        # Wait briefly for the loop to become ready so early audio isn't lost.
        self._loop_ready.wait(timeout=2.0)

    # ---- called by the pipeline worker thread ----

    def run_async(self, inbox) -> None:
        """Pull (kind, samples, sample_rate, pts) tuples from the
        pipeline and forward to the asyncio loop."""
        while True:
            item = inbox.get()
            if item is None:
                # Flush any trailing in-progress sentence so the final
                # utterance isn't lost on shutdown.
                self._flush_live(reason="shutdown")
                self._emit_update(force=True)
                self._stopped.set()
                if self._loop is not None and self._audio_q is not None:
                    # Unblock _send_audio_loop (blocked on queue.get) before
                    # stopping the loop so no tasks are left pending.
                    self._loop.call_soon_threadsafe(self._audio_q.put_nowait, None)
                    self._loop.call_soon_threadsafe(self._loop.stop)
                return
            kind = item[0]
            if kind != "audio":
                continue
            _, samples, sample_rate, _pts = item
            self._handle_audio(samples, sample_rate)

    def _handle_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        if self._loop is None or self._audio_q is None:
            return

        # (n, channels) -> mono float32 at int16 scale.
        if samples.ndim == 2:
            mono = samples.mean(axis=1).astype(np.float32)
        else:
            mono = samples.astype(np.float32)

        # Resample to 16 kHz.
        if sample_rate != 16000:
            if sample_rate == 48000:
                mono = resample_poly(mono, up=1, down=3).astype(np.float32)
            else:
                # General case: approximate rational resample.
                from math import gcd

                g = gcd(sample_rate, 16000)
                up = 16000 // g
                down = sample_rate // g
                mono = resample_poly(mono, up=up, down=down).astype(np.float32)

        # aiortc's OpusDecoder forces format="s16", so samples arrive as
        # int16-scale float32. A float-format build would put values in
        # [-1, 1]; scale up if we see that.
        if mono.size and float(np.max(np.abs(mono))) <= 1.5:
            mono = mono * 32768.0

        chunk_ms = (mono.size / 16000.0) * 1000.0
        self._stream_ms_sent = self._stream_ms_sent + chunk_ms
        with self._clock_lock:
            # Refresh the stream_ms → monotonic anchor on each chunk.
            self._clock_anchor = (self._stream_ms_sent, time.monotonic())

        pcm = np.clip(mono, -32768.0, 32767.0).astype(np.int16).tobytes()

        loop = self._loop
        q = self._audio_q

        def _enqueue():
            try:
                q.put_nowait(pcm)
            except asyncio.QueueFull:
                self._drops += 1
                if self._drops == 1 or self._drops % 200 == 0:
                    logger.warning(
                        "[soniox] audio queue full; dropped %d",
                        self._drops,
                    )

        loop.call_soon_threadsafe(_enqueue)
