"""Minimal IoU face tracker + per-track rolling buffer of face crops.

Frame rate note: LR-ASD expects 25 fps video paired 4:1 with 100 Hz
MFCC. Our glasses stream is 15 fps. To preserve the 4:1 wall-clock
ratio the model was trained on we duplicate frames via 3:5 pulldown
inside the per-track ring, so a full 10-crop window represents ~400 ms
of real time (matched to 40 MFCC steps on the audio side).
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

WINDOW_FRAMES = 20

logger = logging.getLogger(__name__)


@dataclass
class Track:
    tid: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    last_seen: float
    crops: collections.deque = field(default_factory=lambda: collections.deque(maxlen=10))
    phase: float = 0.0  # pulldown accumulator
    # Most recent YuNet detection row (shape (1,15): bbox + 5 landmarks
    # + score). Retained across skip frames so the identity resolver
    # can run alignCrop whenever it likes.
    last_face_row: np.ndarray | None = None
    # Number of tracker.update() calls this track has survived. Used as
    # a stability gate before we commit to an identity (short-lived
    # tracks from camera jitter get filtered out).
    frame_count: int = 0
    # Number of update() calls in which this track was matched to a
    # REAL detection (not just kept alive on a stale bbox). Used as a
    # burn-in gate before exposing the track for inference, so a
    # one-shot YuNet false positive can't reach `ready_tracks` simply
    # by accumulating duplicated crops via the pulldown.
    det_count: int = 0


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _crop_face(frame: np.ndarray, bbox, out_size: int = 112, scale: float = 1.4) -> np.ndarray:
    """Match LR-ASD training preprocessing (DataLoaderTalk.load_visual,
    non-aug branch): expand bbox by `scale`, clip to frame, convert to
    gray, resize directly to (out_size, out_size). No inner center crop —
    that step lives in the upstream Columbia_test.py demo but is absent
    from the training loader, so applying it at inference shrinks the
    effective coverage to ~0.7× of the face box and clips mouth pixels
    on wide/profile faces.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half = max(x2 - x1, y2 - y1) * 0.5 * scale
    nx1 = max(0, int(cx - half))
    ny1 = max(0, int(cy - half))
    nx2 = min(w, int(cx + half))
    ny2 = min(h, int(cy + half))
    if nx2 <= nx1 or ny2 <= ny1:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    roi = frame[ny1:ny2, nx1:nx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


class FaceTracker:
    """IoU-associate new YuNet detections against active tracks. On each
    update, push (possibly duplicated) gray crops into every matched
    track's rolling buffer so we always have a 10-frame window ready."""

    def __init__(
        self,
        *,
        window_frames: int = 10,
        iou_threshold: float = 0.3,
        max_gap_sec: float = 0.7,
        src_fps: float = 15.0,
        target_fps: float = 25.0,
        crop_size: int = 112,
        bbox_alpha: float = 0.5,
        min_det_score: float = 0.7,
        min_det_count: int = 5,
        debug: bool = False,
    ):
        self.window_frames = window_frames
        self.iou_threshold = iou_threshold
        self.max_gap_sec = max_gap_sec
        self.crop_size = crop_size
        # EMA weight on fresh detections: new = alpha*det + (1-alpha)*prev.
        # Kills YuNet's per-frame bbox jitter so the 112×112 crop doesn't
        # wobble while the head is still — which LR-ASD would otherwise
        # read as lip motion.
        self.bbox_alpha = bbox_alpha
        # Drop YuNet detections below this confidence before tracking.
        # YuNet's own gate is 0.6; we tighten to 0.7 so transient false
        # positives (background clutter, hands, posters) never spawn a
        # track that could later be assigned an identity.
        self.min_det_score = min_det_score
        # A track must be matched to a REAL detection at least this many
        # times before it shows up in `ready_tracks`. Pure pulldown
        # duplication can't satisfy the gate, so a one-shot detection
        # can't drive ASD inference or identity resolution.
        self.min_det_count = min_det_count
        self.debug = debug
        self.pulldown_ratio = target_fps / src_fps  # 25/15 ≈ 1.667
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def _new_track(self, bbox, now: float) -> Track:
        t = Track(
            tid=self._next_id,
            bbox=bbox,
            last_seen=now,
            crops=collections.deque(maxlen=self.window_frames),
        )
        self._next_id += 1
        return t

    def update(
        self,
        frame: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], np.ndarray]] | None,
    ) -> list[Track]:
        """detections: list of (x1,y1,x2,y2) or None if detect skipped this frame.
        Returns the list of currently-active tracks.

        A track is only kept alive (last_seen refreshed) when it is matched
        to a REAL detection. Skip frames still push crops into the rolling
        window so inference stays fresh, but can't keep dead tracks alive.
        """
        now = time.monotonic()

        # Map tid → bbox to use for this frame's crop (real det or last known).
        frame_bbox: dict[int, tuple[int, int, int, int]] = {}
        # Subset of above whose bbox came from an actual detection this frame.
        really_matched: set[int] = set()

        # Per-frame face-row for tracks matched to a real detection this
        # frame (used by the identity resolver for SFace alignCrop).
        frame_face_row: dict[int, np.ndarray] = {}

        if detections is None:
            for tid, t in self._tracks.items():
                frame_bbox[tid] = t.bbox
        else:
            # Drop low-confidence detections before they can spawn a
            # track. YuNet's last column is the per-detection score.
            filtered: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
            for det_bbox, det_row in detections:
                row = np.asarray(det_row, dtype=np.float32).reshape(-1)
                if row.size >= 1 and float(row[-1]) < self.min_det_score:
                    if self.debug:
                        logger.debug(
                            "[asd-track] drop low-conf det score=%.2f bbox=%s",
                            float(row[-1]),
                            det_bbox,
                        )
                    continue
                filtered.append((det_bbox, det_row))
            detections = filtered
            used_tracks: set[int] = set()
            for det_bbox, det_row in detections:
                best_tid, best_iou = None, 0.0
                for tid, t in self._tracks.items():
                    if tid in used_tracks:
                        continue
                    iou = _iou(det_bbox, t.bbox)
                    if iou > best_iou:
                        best_iou, best_tid = iou, tid
                if best_tid is not None and best_iou >= self.iou_threshold:
                    prev = self._tracks[best_tid].bbox
                    a = self.bbox_alpha
                    smoothed = (
                        int(a * det_bbox[0] + (1 - a) * prev[0]),
                        int(a * det_bbox[1] + (1 - a) * prev[1]),
                        int(a * det_bbox[2] + (1 - a) * prev[2]),
                        int(a * det_bbox[3] + (1 - a) * prev[3]),
                    )
                    frame_bbox[best_tid] = smoothed
                    frame_face_row[best_tid] = det_row
                    really_matched.add(best_tid)
                    used_tracks.add(best_tid)
                else:
                    t = self._new_track(det_bbox, now)
                    self._tracks[t.tid] = t
                    frame_bbox[t.tid] = det_bbox
                    frame_face_row[t.tid] = det_row
                    really_matched.add(t.tid)
                    if self.debug:
                        logger.debug(
                            "[asd-track] new #%d bbox=%s best_iou=%.2f",
                            t.tid,
                            det_bbox,
                            best_iou,
                        )
            # Unmatched tracks this detection frame: keep their last bbox
            # for the crop (so the buffer stays continuous until drop),
            # but do NOT mark as seen.
            for tid, t in self._tracks.items():
                if tid not in frame_bbox:
                    frame_bbox[tid] = t.bbox

        for tid in list(self._tracks.keys()):
            t = self._tracks[tid]
            if tid in really_matched:
                t.bbox = frame_bbox[tid]
                t.last_seen = now
                t.det_count += 1
                if tid in frame_face_row:
                    t.last_face_row = frame_face_row[tid]
            if now - t.last_seen > self.max_gap_sec:
                if self.debug:
                    logger.debug(
                        "[asd-track] drop #%d (gap %.2fs)",
                        tid,
                        now - t.last_seen,
                    )
                del self._tracks[tid]
                continue
            # Still alive → push a crop (matched bbox if detected, else stale bbox).
            crop = _crop_face(frame, frame_bbox[tid], out_size=self.crop_size)
            prev_phase = t.phase
            t.phase = prev_phase + self.pulldown_ratio
            copies = int(t.phase) - int(prev_phase)
            for _ in range(max(1, copies)):
                t.crops.append(crop)
            t.frame_count += 1

        return list(self._tracks.values())

    def ready_tracks(self) -> list[Track]:
        return [
            t
            for t in self._tracks.values()
            if len(t.crops) == self.window_frames and t.det_count >= self.min_det_count
        ]
