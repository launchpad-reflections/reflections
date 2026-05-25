"""Overlay drawing (green/red/white boxes)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .track import Track


def draw_tracks(
    frame: np.ndarray,
    live_tracks: list[Track],
    scores_snapshot: dict[int, tuple[float, float]],
    identities_snapshot: dict[int, str],
    *,
    score_threshold: float,
    display_timeout_sec: float,
) -> np.ndarray:
    """Pull live bboxes straight from the tracker so boxes track faces in
    real time, independent of inference latency."""
    now = time.monotonic()
    for t in live_tracks:
        x1, y1, x2, y2 = t.bbox
        score_entry = scores_snapshot.get(t.tid)
        if score_entry is None or (now - score_entry[1]) > display_timeout_sec:
            # No fresh score yet — draw a faint white box so the user sees
            # the face is detected while we wait.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
            continue
        score, _ts = score_entry
        color = (0, 255, 0) if score > score_threshold else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = identities_snapshot.get(t.tid)
        label = f"{name} {score:+.2f}" if name is not None else f"{score:+.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame
