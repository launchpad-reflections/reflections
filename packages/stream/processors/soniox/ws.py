"""WebSocket transport to the Soniox real-time transcription API."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

logger = logging.getLogger(__name__)


class SonioxWebSocketMixin:
    """Persistent Soniox WebSocket session with reconnect."""

    api_key: str
    model: str
    on_transcript: Callable[[str, str | None], None] | None
    on_interim: Callable[[str], None] | None
    _stopped: threading.Event
    _loop: asyncio.AbstractEventLoop | None
    _audio_q: asyncio.Queue | None
    _loop_ready: threading.Event
    _last_final_end_ms: int
    _stream_ms_sent: float
    _clock_lock: threading.Lock
    _clock_anchor: tuple[float, float] | None

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._audio_q = asyncio.Queue(maxsize=2000)
        self._loop_ready.set()
        try:
            self._loop.run_until_complete(self._run_forever())
        except Exception as e:
            logger.error("[soniox] loop crashed: %s", e)
        finally:
            # Cancel any tasks left pending by loop.stop() and drain them
            # so Python doesn't warn "Task was destroyed but it is pending!".
            try:
                pending = asyncio.all_tasks(self._loop)
                if pending:
                    for t in pending:
                        t.cancel()
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                self._loop.close()
            except Exception:
                pass

    async def _run_forever(self) -> None:
        import websockets

        while not self._stopped.is_set():
            try:
                async with websockets.connect(
                    SONIOX_WS_URL,
                    ping_interval=None,
                    max_size=2**22,
                ) as ws:
                    config = {
                        "api_key": self.api_key,
                        "model": self.model,
                        "audio_format": "pcm_s16le",
                        "sample_rate": 16000,
                        "num_channels": 1,
                        "enable_language_identification": True,
                        "enable_endpoint_detection": True,
                        # Soniox stt-rt-v4 attaches a per-token "speaker"
                        # field once this is on. We use it as the
                        # authoritative diarization signal and fuse it
                        # with face-side ASD via _classify_speaker.
                        "enable_speaker_diarization": True,
                        "max_non_final_tokens_duration_ms": 700,
                        "language_hints": ["en"],
                    }
                    await ws.send(json.dumps(config))
                    logger.info("[soniox] connected")

                    self._last_final_end_ms = 0
                    # Soniox restarts its stream timeline at 0 on each
                    # new session — reset our parallel clock.
                    self._stream_ms_sent = 0.0
                    with self._clock_lock:
                        self._clock_anchor = None
                    # Speaker numbering also restarts at "1" on each
                    # session, so any prior spk_id → identity bindings
                    # are now meaningless. Clear them; the fusion layer
                    # will rebuild as soon as new finals start arriving.
                    self._clear_fusion_state()
                    # Drop any backlog queued before connect so we don't
                    # burst stale audio.
                    while not self._audio_q.empty():
                        try:
                            self._audio_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    send_task = asyncio.create_task(self._send_audio_loop(ws))
                    recv_task = asyncio.create_task(self._recv_loop(ws))
                    ka_task = asyncio.create_task(self._keepalive_loop(ws))
                    dwell_task = asyncio.create_task(self._dwell_loop())
                    all_tasks = {send_task, recv_task, ka_task, dwell_task}
                    try:
                        done, pending = await asyncio.wait(
                            all_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()
                        for t in pending:
                            try:
                                await t
                            except (asyncio.CancelledError, Exception):
                                pass
                        for t in done:
                            exc = t.exception()
                            if exc:
                                raise exc
                    except BaseException:
                        for t in all_tasks:
                            if not t.done():
                                try:
                                    t.cancel()
                                except Exception:
                                    pass
                        raise
            except Exception as e:
                if self._stopped.is_set():
                    return
                logger.warning("[soniox] session error: %s; reconnecting in 1s", e)
                await asyncio.sleep(1.0)

    async def _send_audio_loop(self, ws) -> None:
        while not self._stopped.is_set():
            chunk = await self._audio_q.get()
            if chunk is None:
                return
            await ws.send(chunk)

    async def _keepalive_loop(self, ws) -> None:
        while not self._stopped.is_set():
            await asyncio.sleep(15)
            await ws.send(json.dumps({"type": "keepalive"}))

    async def _recv_loop(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            err = data.get("error_code") or data.get("error")
            if err:
                logger.error("[soniox] server error: %s", data)
                return

            tokens = data.get("tokens") or []
            if not tokens:
                continue

            # Non-final tokens → current interim tail. Filter out the
            # Soniox endpoint marker "<end>" if it ever leaks through as
            # non-final; it's a control token, not transcript text.
            interim = "".join(
                t.get("text", "")
                for t in tokens
                if not t.get("is_final") and t.get("text") != "<end>"
            ).strip()

            if self.on_interim is not None:
                try:
                    self.on_interim(interim)
                except Exception as e:
                    logger.error("[soniox] on_interim raised: %s", e)

            new_finals = [
                t
                for t in tokens
                if t.get("is_final") and int(t.get("end_ms", 0)) > self._last_final_end_ms
            ]

            # Plain text for the user-facing on_transcript callback
            # (excluding the endpoint marker), and separate flag for the
            # endpoint token itself — it's a strong sentence boundary
            # signal from Soniox's endpoint detector.
            new_final_text_visible = "".join(
                t.get("text", "") for t in new_finals if t.get("text") != "<end>"
            )
            endpoint_hit = any(t.get("text") == "<end>" for t in new_finals)

            if new_finals:
                self._last_final_end_ms = max(int(t.get("end_ms", 0)) for t in new_finals)

            # Group new finals into RUNS of consecutive same-speaker
            # tokens. With diarization on, a single Soniox message can
            # carry tokens from multiple speakers; coalescing them under
            # one speaker label would silently mis-attribute speech.
            visible_finals = [t for t in new_finals if t.get("text") != "<end>"]

            def _spk_of(tok: dict) -> str | None:
                s = tok.get("speaker")
                if s is None:
                    return None
                s = str(s).strip()
                if not s or s == "0":
                    return None
                return s

            runs: list[tuple[str | None, list[dict]]] = []
            for tok in visible_finals:
                spk = _spk_of(tok)
                if runs and runs[-1][0] == spk:
                    runs[-1][1].append(tok)
                else:
                    runs.append((spk, [tok]))

            # Diagnostic: show the raw speaker tags Soniox sent for
            # this batch of new finals. If every entry is None, Soniox
            # isn't tagging speakers (account/plan issue or v4 quirk);
            # if entries vary across segments, diarization is working
            # and any mis-attribution is a fusion-weight problem.
            if visible_finals:
                raw_tags = [tok.get("speaker") for tok in visible_finals]
                logger.debug(
                    "[soniox-diag] raw_speaker_tags=%s runs=%s",
                    raw_tags,
                    [
                        (s, "".join(t.get("text", "") for t in toks).strip()[:30])
                        for s, toks in runs
                    ],
                )

            # Use the LAST run's speaker as the message-level
            # final_speaker for the live-state machine — it represents
            # who is currently talking, which is what _ingest_tokens
            # uses to drive the speaker-flip logic.
            final_speaker: str | None = None

            for spk, run_tokens in runs:
                run_text = "".join(t.get("text", "") for t in run_tokens)
                if not run_text.strip():
                    continue
                run_start_ms = min(int(t.get("start_ms", 0)) for t in run_tokens)
                run_end_ms = max(int(t.get("end_ms", 0)) for t in run_tokens)
                run_speaker = self._classify_speaker(run_start_ms, run_end_ms, spk)
                final_speaker = run_speaker

                if self.on_transcript is not None:
                    try:
                        self.on_transcript(run_text.strip(), run_speaker)
                    except Exception as e:
                        logger.error("[soniox] on_transcript raised: %s", e)

            self._ingest_tokens(
                new_final_text=new_final_text_visible,
                interim=interim,
                final_speaker=final_speaker,
                endpoint_hit=endpoint_hit,
            )
