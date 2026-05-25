"""LR-ASD batch inference worker."""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from python_speech_features import mfcc as compute_mfcc

from .track import WINDOW_FRAMES

if TYPE_CHECKING:
    from . import ASDProcessor

T_AUDIO = WINDOW_FRAMES * 4  # 80
SAMPLES_FOR_MFCC = (T_AUDIO - 1) * 160 + 400  # 13040 samples ~= 815 ms
AUDIO_BUFFER_SAMPLES = 24000  # ~1.5 s ring (must exceed SAMPLES_FOR_MFCC with headroom)

# Audio-energy gate. Below this dBFS we treat the audio as silent and skip
# both LR-ASD inference (saves CPU) and retroactive who_spoke history
# samples (silent-window scores can't drive misattribution). 0 dBFS is
# int16 max; resting noise floor on aiortc 16 kHz typically sits near
# -55 dBFS, conversational speech peaks -25 to -15 dBFS, so -45 dBFS
# leaves a clean margin while still passing whispered speech.
AUDIO_GATE_DBFS = -45.0

logger = logging.getLogger(__name__)


def current_mfcc(audio_buf: np.ndarray, audio_lock: threading.Lock) -> np.ndarray | None:
    with audio_lock:
        if audio_buf.size < SAMPLES_FOR_MFCC:
            return None
        audio = audio_buf[-SAMPLES_FOR_MFCC:].copy()
    try:
        feats = compute_mfcc(
            audio,
            samplerate=16000,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        ).astype(np.float32)
    except Exception as e:
        logger.error("[asd] mfcc error: %s", e)
        return None
    if feats.shape[0] >= T_AUDIO:
        return feats[-T_AUDIO:]
    pad = np.zeros((T_AUDIO - feats.shape[0], 13), dtype=np.float32)
    return np.concatenate([pad, feats], axis=0)


def window_peak_dbfs(
    dbfs_history: collections.deque[tuple[float, float]],
    audio_lock: threading.Lock,
    *,
    infer_window_lo: float,
) -> float:
    with audio_lock:
        return max(
            (dbfs for ts, dbfs in dbfs_history if ts >= infer_window_lo),
            default=-120.0,
        )


class InferWorker:
    """Background LR-ASD forward pass fed by a size-1 work queue."""

    def __init__(self, processor: ASDProcessor) -> None:
        self._processor = processor
        self._infer_q: queue.Queue = queue.Queue(maxsize=1)
        self._infer_stop = threading.Event()
        self._infer_thread = threading.Thread(
            target=self._infer_loop, name="asd-infer", daemon=True
        )

    def start(self) -> None:
        self._infer_thread.start()

    def stop(self) -> None:
        self._infer_stop.set()
        try:
            self._infer_q.put_nowait(None)
        except queue.Full:
            pass

    def submit(self, payload: tuple) -> None:
        try:
            self._infer_q.put_nowait(payload)
        except queue.Full:
            try:
                self._infer_q.get_nowait()
                self._infer_q.put_nowait(payload)
            except (queue.Empty, queue.Full):
                pass

    def _infer_loop(self) -> None:
        # Lazy import so torch/weights load on worker startup, not on
        # pipeline startup (keeps render thread snappy at boot).
        from third_party.lr_asd import load, predict

        proc = self._processor
        try:
            load()
        except Exception as e:
            logger.error("[asd] model load failed: %s", e)
            return

        while not self._infer_stop.is_set():
            try:
                item = self._infer_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                return
            tids, video_batch, audio_batch, frame_bgr, face_rows, frame_counts = item
            try:
                t0 = time.monotonic()
                logits = predict(video_batch, audio_batch)  # (N, T_video)
                infer_ms = (time.monotonic() - t0) * 1000
            except Exception as e:
                logger.error("[asd] predict error: %s", e)
                continue

            # Max (not mean) over the recent tail: short utterances (1–2
            # words) produce only a few speech-frames inside the window;
            # averaging dilutes them with surrounding silence so the
            # pooled score fails to cross threshold. Max preserves the
            # peak, which is what we actually care about for "did this
            # person speak in the last ~240 ms?".
            scores = logits[:, -6:].max(axis=1)
            ts = time.monotonic()
            # Prune state for tracks that have died since the last
            # inference. Without this, _history retains scores for
            # dropped tracks and who_spoke can credit a face that
            # already left frame. (The render thread also prunes via
            # draw(), but who_spoke can be queried between draws.)
            # list() snapshots dict.values() atomically — the tracker can
            # mutate _tracks from the render thread while we iterate.
            live_tids = {t.tid for t in list(proc._tracker._tracks.values())}
            with proc._scores_lock:
                for dead_tid in list(proc._scores.keys()):
                    if dead_tid not in live_tids:
                        del proc._scores[dead_tid]
                for dead_tid in list(proc._identities.keys()):
                    if dead_tid not in live_tids:
                        del proc._identities[dead_tid]
                if proc._history and len(live_tids) < len({h[0] for h in proc._history}):
                    proc._history = collections.deque(
                        ((tid, score, t) for (tid, score, t) in proc._history if tid in live_tids),
                        maxlen=proc._history.maxlen,
                    )
                for tid, s in zip(tids, scores):
                    raw = float(s)
                    prev = proc._scores.get(tid)
                    ema = (
                        proc.ema_alpha * raw + (1 - proc.ema_alpha) * prev[0]
                        if prev is not None
                        else raw
                    )
                    proc._scores[tid] = (ema, ts)
                    # History stores the RAW per-inference score (not
                    # EMA-smoothed) so retroactive lookups see the real
                    # signal for a given moment.
                    proc._history.append((tid, raw, ts))

            # Identity resolution — only for tracks the model thinks are
            # actively speaking this batch. This matches the design
            # requirement that we only embed faces while they are
            # talking (so the gallery gets real speaker frames, not
            # random listener frames).
            if proc._identity is not None:
                for tid, s in zip(tids, scores):
                    if float(s) <= proc.score_threshold:
                        continue
                    # Stability gate: skip short-lived tracks (camera
                    # jitter / false positives). Without this, transient
                    # tracks get matched by luck against existing gallery
                    # entries and pollute the identity assignments.
                    if frame_counts.get(tid, 0) < proc.min_frames_for_identity:
                        continue
                    with proc._scores_lock:
                        if tid in proc._identities:
                            continue
                    row = face_rows.get(tid)
                    if row is None or frame_bgr is None:
                        continue
                    try:
                        name = proc._identity.resolve(frame_bgr, row, tid)
                    except Exception as e:
                        logger.error("[asd] identity resolve error: %s", e)
                        name = None
                    if name is None:
                        continue
                    with proc._scores_lock:
                        proc._identities[tid] = name

            if proc.debug:
                logger.debug(
                    "[asd] infer %.1fms tracks=%d scores=%s",
                    infer_ms,
                    len(tids),
                    [f"{float(s):+.2f}" for s in scores],
                )
