"""Active speaker detection via LR-ASD.

Detection, tracking, and audio buffering run synchronously on the
render thread (YuNet is ~25 ms, runs every Nth frame; audio append is
trivial). Only the LR-ASD forward pass — the expensive step — runs on
a background worker thread fed by a size-1 queue that always holds the
latest work request, so the worker never processes stale frames.

draw() reads bboxes directly from the live tracker, so boxes move with
faces in real time. Scores are updated by the worker as they come in.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from math import gcd

import numpy as np
from reflections.config import REPO_ROOT
from scipy.signal import resample_poly

from stream.pipeline import Processor

from .._identity import IdentityResolver
from .detect import FaceDetector
from .draw import draw_tracks
from .infer import (
    AUDIO_BUFFER_SAMPLES,
    AUDIO_GATE_DBFS,
    T_AUDIO,
    InferWorker,
    current_mfcc,
    window_peak_dbfs,
)
from .track import WINDOW_FRAMES, FaceTracker

FACE_REC_PATH = REPO_ROOT / "models" / "w600k_mbf.onnx"
GALLERY_NPZ = REPO_ROOT / "models" / "face_gallery.npz"
GALLERY_JSON = REPO_ROOT / "models" / "face_gallery.json"

# Tolerance window when matching a who_spoke history sample timestamp
# against the dBFS history; covers chunk-cadence jitter.
AUDIO_GATE_LOOKUP_PAD_S = 0.15

logger = logging.getLogger(__name__)


class ASDProcessor(Processor):
    # Sync: detection + tracking + audio buffering run on the render
    # thread. The slow forward pass runs on an internal worker thread.
    mode = "sync"
    name = "asd"
    consumes = frozenset({"video", "audio"})

    def __init__(
        self,
        *,
        detect_every_n: int = 1,
        inference_hz: float = 10.0,
        score_threshold: float = -0.5,
        ema_alpha: float = 0.6,
        display_timeout_sec: float = 0.5,
        src_fps: float = 15.0,
        enable_identity: bool = True,
        min_frames_for_identity: int = 30,
        debug: bool = False,
    ):
        self.detect_every_n = detect_every_n
        self.min_infer_interval = 1.0 / max(inference_hz, 0.1)
        self.score_threshold = score_threshold
        self.ema_alpha = ema_alpha
        self.display_timeout_sec = display_timeout_sec
        self.min_frames_for_identity = min_frames_for_identity
        self.debug = debug

        self._tracker = FaceTracker(
            window_frames=WINDOW_FRAMES,
            src_fps=src_fps,
            target_fps=25.0,
            debug=debug,
        )
        self._detector = FaceDetector()
        self._frame_idx = 0
        self._last_infer_ts = 0.0

        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._audio_lock = threading.Lock()
        # (mono_ts, dbfs) per ingested audio chunk. ~200 entries covers
        # roughly 2 s at our typical 100 Hz chunk cadence — enough to
        # answer "was the mic active around time T?" for any inference
        # window or retroactive who_spoke lookup we care about.
        self._dbfs_history: collections.deque[tuple[float, float]] = collections.deque(maxlen=200)
        self._recent_dbfs: float = -120.0

        # tid → (score_ema, ts)
        self._scores: dict[int, tuple[float, float]] = {}
        # Rolling history of per-track per-inference samples used for
        # retroactive speaker attribution. ~60 s at 10 Hz × ≤3 tracks.
        self._history: collections.deque[tuple[int, float, float]] = collections.deque(maxlen=2000)
        self._scores_lock = threading.Lock()

        # tid -> resolved identity name (e.g. "Bob", "Person 3")
        self._identities: dict[int, str] = {}

        self._identity: IdentityResolver | None = None
        if enable_identity:
            try:
                self._identity = IdentityResolver(
                    model_path=FACE_REC_PATH,
                    gallery_path=GALLERY_NPZ,
                    index_path=GALLERY_JSON,
                    debug=debug,
                )
            except FileNotFoundError as e:
                logger.warning("[asd] identity disabled: %s", e)
                self._identity = None
            except Exception as e:
                logger.error("[asd] identity init failed: %s", e)
                self._identity = None

        self._infer_worker = InferWorker(self)
        self._infer_worker.start()

    # ---- sync callbacks on render thread ----

    def on_audio(self, samples: np.ndarray, sample_rate: int, pts: float) -> None:
        if samples.ndim == 2:
            mono = samples.mean(axis=1).astype(np.float32)
        else:
            mono = samples.astype(np.float32)

        if sample_rate != 16000:
            if sample_rate == 48000:
                mono = resample_poly(mono, up=1, down=3).astype(np.float32)
            else:
                g = gcd(sample_rate, 16000)
                mono = resample_poly(mono, up=16000 // g, down=sample_rate // g).astype(np.float32)

        # LR-ASD was trained with MFCC computed on int16-scale audio
        # (scipy wavfile.read output). Keep values at int16 scale; if
        # aiortc gave us [-1, 1] floats (float-format decoder build),
        # scale up.
        if mono.size and float(np.max(np.abs(mono))) <= 1.5:
            mono = mono * 32768.0

        # dBFS for this chunk relative to int16 max. Used to gate
        # inference and to filter retroactive history lookups so silent
        # frames can't drive speaker misattribution.
        if mono.size:
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
            dbfs = 20.0 * float(np.log10(rms / 32768.0 + 1e-9))
        else:
            dbfs = -120.0
        ts_now = time.monotonic()

        with self._audio_lock:
            self._audio_buf = np.concatenate([self._audio_buf, mono])
            if self._audio_buf.size > AUDIO_BUFFER_SAMPLES:
                self._audio_buf = self._audio_buf[-AUDIO_BUFFER_SAMPLES:]
            self._dbfs_history.append((ts_now, dbfs))
            self._recent_dbfs = dbfs

    def on_video(self, frame: np.ndarray, pts: float) -> None:
        self._frame_idx += 1
        detect_now = ((self._frame_idx - 1) % self.detect_every_n) == 0
        try:
            dets = self._detector.detect(frame) if detect_now else None
        except Exception as e:
            logger.error("[asd] detect error: %s", e)
            dets = None
        if self.debug and dets is not None:
            logger.debug(
                "[asd] frame#%d dets=%d tracks_before=%d",
                self._frame_idx,
                len(dets),
                len(self._tracker._tracks),
            )
        self._tracker.update(frame, dets)

        now = time.monotonic()
        if now - self._last_infer_ts < self.min_infer_interval:
            return

        ready = self._tracker.ready_tracks()
        if not ready:
            return

        # Audio-energy gate: if the mic has been silent across the
        # entire inference window, skip the LR-ASD forward pass. Looks
        # at the dBFS history covering the last ~400 ms (the visual
        # context length each inference summarizes) — if the loudest
        # chunk in that window didn't cross AUDIO_GATE_DBFS we know
        # nothing was actually being said and any positive score would
        # be a lip-only false positive.
        infer_window_lo = now - 0.4
        window_peak = window_peak_dbfs(
            self._dbfs_history,
            self._audio_lock,
            infer_window_lo=infer_window_lo,
        )
        if window_peak < AUDIO_GATE_DBFS:
            if self.debug:
                logger.debug(
                    "[asd] audio silent (peak %.1f dBFS); " "skipping infer for %d track(s)",
                    window_peak,
                    len(ready),
                )
            return

        mfcc = current_mfcc(self._audio_buf, self._audio_lock)
        if mfcc is None:
            return

        video_batch = np.stack([np.stack(list(t.crops), axis=0) for t in ready], axis=0).astype(
            np.float32
        )
        audio_batch = np.broadcast_to(mfcc[None, :, :], (len(ready), T_AUDIO, 13)).astype(
            np.float32
        )
        tids = [t.tid for t in ready]
        # Per-tid face row snapshot (for SFace alignCrop on worker).
        # May be None for a track if it hasn't received a real detection
        # yet (unlikely for ready tracks, but we guard anyway).
        face_rows = {t.tid: t.last_face_row for t in ready}
        frame_counts = {t.tid: t.frame_count for t in ready}
        # Send the original BGR frame (read-only snapshot) so the worker
        # can alignCrop. A bare reference is fine — numpy arrays are not
        # mutated after dispatch.
        frame_snapshot = frame

        payload = (
            tids,
            video_batch,
            audio_batch,
            frame_snapshot,
            face_rows,
            frame_counts,
        )

        # Hand off to worker. If something's already queued, drop it —
        # freshness matters more than completeness.
        self._infer_worker.submit(payload)
        self._last_infer_ts = now

    # ---- retroactive speaker attribution ----

    def _was_audio_active(self, ts: float) -> bool:
        """True iff at least one dBFS sample within ±AUDIO_GATE_LOOKUP_PAD_S
        of `ts` exceeded AUDIO_GATE_DBFS. Used to filter score-history
        samples so silent windows can't drive who_spoke decisions even
        if a stale ASD score happened to land there."""
        with self._audio_lock:
            for sample_ts, dbfs in self._dbfs_history:
                if abs(sample_ts - ts) <= AUDIO_GATE_LOOKUP_PAD_S and dbfs >= AUDIO_GATE_DBFS:
                    return True
        return False

    def who_spoke(
        self,
        t_start: float,
        t_end: float,
        *,
        threshold: float | None = None,
        min_samples: int = 2,
        min_positive_ratio: float = 0.4,
    ) -> int | None:
        """Return the track id that spoke most during monotonic window
        [t_start, t_end], or None if no track crossed threshold there.

        Works off the raw per-inference history, so finds the speaker
        for time ranges that have already passed.

        Guards against false attribution to a face that wasn't actually
        speaking by requiring (a) the track still exists in the live
        tracker — dead tracks can't be credited, (b) at least
        `min_samples` positive ASD samples for the candidate, AND
        (c) those positives represent at least `min_positive_ratio` of
        the track's samples in the window. Without (c), two flicker
        frames in a multi-second window could outvote zero positives
        for everyone else."""
        if t_end <= t_start:
            return None
        thresh = self.score_threshold if threshold is None else threshold
        # Accept samples slightly outside the window to account for the
        # ~400 ms visual context each inference summarizes.
        pad = 0.3
        lo, hi = t_start - pad, t_end + pad
        # Liveness gate: a track that's no longer in the tracker can't
        # have been the speaker now. Without this, stale ASD scores
        # from a face that left the frame moments ago can still win
        # who_spoke and steal the wearer's audio.
        # list() snapshots dict.values() atomically — the tracker can
        # mutate _tracks from the render thread while we iterate.
        live_tids = {t.tid for t in list(self._tracker._tracks.values())}
        if not live_tids:
            return None
        with self._scores_lock:
            history_snapshot = list(self._history)
        snapshot = [
            (tid, score, ts)
            for (tid, score, ts) in history_snapshot
            if lo <= ts <= hi and tid in live_tids
        ]
        # Drop samples whose timestamp falls inside an audio-silent
        # window — without this filter, a stray positive score during
        # silence could still win the attribution.
        snapshot = [(tid, score) for (tid, score, ts) in snapshot if self._was_audio_active(ts)]
        if not snapshot:
            return None
        positive_by_tid: dict[int, int] = {}
        total_by_tid: dict[int, int] = {}
        for tid, score in snapshot:
            total_by_tid[tid] = total_by_tid.get(tid, 0) + 1
            if score > thresh:
                positive_by_tid[tid] = positive_by_tid.get(tid, 0) + 1
        if not positive_by_tid:
            return None
        # Winner = most positive samples, subject to absolute count and
        # ratio gates. Ratio prevents a 2-out-of-12 flicker from
        # crediting a track that mostly stayed silent during the window.
        best_tid, best_pos = max(positive_by_tid.items(), key=lambda kv: kv[1])
        if best_pos < min_samples:
            return None
        total = total_by_tid.get(best_tid, 0)
        if total > 0 and (best_pos / total) < min_positive_ratio:
            return None
        return best_tid

    def who_is_speaking_now(self, window_s: float = 0.5) -> str | None:
        """Identity of whoever is speaking right now, or None. Thin
        wrapper over who_spoke_name looking at the last `window_s` of
        ASD history.

        The window is wider than it used to be (was 0.3 s) so we get
        more samples per call — at 10 Hz inference that's up to 5
        samples, enough for the ratio gate in who_spoke to work. We
        require min_samples=2 to avoid one stray positive labeling a
        face as 'currently speaking' and stealing audio attribution."""
        t_end = time.monotonic()
        return self.who_spoke_name(t_end - window_s, t_end, min_samples=2)

    def who_spoke_name(
        self,
        t_start: float,
        t_end: float,
        *,
        threshold: float | None = None,
        min_samples: int = 2,
        min_positive_ratio: float = 0.4,
        prefer_named: bool = True,
    ) -> str | None:
        """Like who_spoke, but returns an identity label ('Bob' or
        'Person 3') if the speaking track has been resolved.

        With `prefer_named=True` (default), unresolved speaking tracks
        return None — callers can then fall back to a sensible default
        (typically the wearer) instead of leaking a bare 'Track N' into
        the transcript. Pass `prefer_named=False` to keep the raw
        `Track N` label for diagnostics."""
        tid = self.who_spoke(
            t_start,
            t_end,
            threshold=threshold,
            min_samples=min_samples,
            min_positive_ratio=min_positive_ratio,
        )
        if tid is None:
            return None
        with self._scores_lock:
            name = self._identities.get(tid)
        if name is not None:
            return name
        return None if prefer_named else f"Track {tid}"

    def rename_identity(self, old_name: str, new_name: str) -> bool:
        """Rename a gallery identity before flush. Returns True if found."""
        if self._identity is None:
            return False
        return self._identity.rename_identity(old_name, new_name)

    def confirm_identity_for_name(self, name: str) -> bool:
        """Promote a pending gallery entry into the permanent gallery.

        Called by the Soniox fusion layer once it has locked a Soniox
        speaker id to `name` — at that point we know audible speech
        actually came from the bound face, so the embedding is worth
        persisting. No-op if the name is already permanent or has no
        pending entry."""
        if self._identity is None:
            return False
        try:
            return self._identity.promote_pending(name)
        except Exception as e:
            logger.error("[asd] promote_pending(%r) failed: %s", name, e)
            return False

    def close(self) -> None:
        """Flush the face gallery to disk (call on shutdown)."""
        self._infer_worker.stop()
        if self._identity is not None:
            self._identity.flush()

    # ---- render thread ----

    def draw(self, frame: np.ndarray) -> np.ndarray:
        live_tracks = list(self._tracker._tracks.values())
        live_ids = {t.tid for t in live_tracks}

        with self._scores_lock:
            # Prune scores + identity cache for tracks that no longer exist.
            for tid in list(self._scores.keys()):
                if tid not in live_ids:
                    del self._scores[tid]
            for tid in list(self._identities.keys()):
                if tid not in live_ids:
                    del self._identities[tid]
            scores_snapshot = dict(self._scores)
            identities_snapshot = dict(self._identities)

        return draw_tracks(
            frame,
            live_tracks,
            scores_snapshot,
            identities_snapshot,
            score_threshold=self.score_threshold,
            display_timeout_sec=self.display_timeout_sec,
        )


__all__ = ["ASDProcessor"]
