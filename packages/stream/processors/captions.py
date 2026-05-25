"""Live transcript caption overlay.

Draws the last few transcript lines at the bottom of the video frame
so captions appear on top of the ASD bboxes, in near-real-time.

Zero-latency design: transcript lines are pre-formatted on the Soniox
thread whenever `update_transcript` is called, and the render thread
just reads a tiny locked snapshot each frame. No image copies, no
alpha blending, no per-frame text measurement.
"""

from __future__ import annotations

import hashlib
import threading

import cv2
import numpy as np

from stream.pipeline import Processor

# BGR tuples for speaker label coloring. Wearer/user is white; other
# speakers cycle through a small palette keyed off label string hash so
# the same person gets a stable color within a session.
_SPEAKER_PALETTE = [
    (120, 255, 120),  # green
    (120, 220, 255),  # amber
    (255, 180, 120),  # blue-ish
    (220, 140, 255),  # magenta
    (255, 255, 120),  # cyan-yellow
]
_USER_COLOR = (235, 235, 235)


def _color_for(label: str) -> tuple[int, int, int]:
    if not label or label == "User":
        return _USER_COLOR
    digest = int(hashlib.md5(label.encode()).hexdigest()[:8], 16)
    return _SPEAKER_PALETTE[digest % len(_SPEAKER_PALETTE)]


class CaptionProcessor(Processor):
    mode = "sync"
    name = "captions"
    # Empty consumes → pipeline skips dispatch, only `draw` is called.
    consumes = frozenset()

    def __init__(
        self,
        *,
        user_name: str = "User",
        max_lines: int = 4,
        font_scale: float = 0.55,
        line_height: int = 22,
        margin: int = 10,
        max_chars_per_line: int = 90,
    ):
        self.user_name = user_name
        self.max_lines = max_lines
        self.font_scale = font_scale
        self.line_height = line_height
        self.margin = margin
        self.max_chars_per_line = max_chars_per_line

        self._lock = threading.Lock()
        # Pre-formatted (label, label_color, text) triples. Built once per
        # transcript update on the Soniox thread — draw() just reads.
        self._lines: list[tuple[str, tuple[int, int, int], str]] = []

    def update_transcript(
        self,
        transcript: list[tuple[str | None, str]],
    ) -> None:
        """Called from Soniox's on_transcript_update callback. Always
        runs on the Soniox asyncio thread, so the lock protects against
        the render thread reading a half-written list."""
        recent = transcript[-self.max_lines :]
        formatted: list[tuple[str, tuple[int, int, int], str]] = []
        for spk, text in recent:
            display = spk if (spk and spk != "User") else self.user_name
            color = _color_for(spk if spk else "User")
            if len(text) > self.max_chars_per_line:
                text = "…" + text[-(self.max_chars_per_line - 1) :]
            formatted.append((display, color, text))
        with self._lock:
            self._lines = formatted

    def clear(self) -> None:
        with self._lock:
            self._lines = []

    def draw(self, frame: np.ndarray) -> np.ndarray:
        # Snapshot under the lock then release — all drawing is lock-free.
        with self._lock:
            lines = list(self._lines)
        if not lines:
            return frame

        h, w = frame.shape[:2]
        n = len(lines)

        # Single background rect behind all lines — one draw call total
        # for the backdrop instead of one per line.
        bg_top = h - self.margin - n * self.line_height - 8
        bg_bottom = h - self.margin + 4
        bg_left = self.margin - 6
        bg_right = w - self.margin + 6
        cv2.rectangle(
            frame,
            (bg_left, bg_top),
            (bg_right, bg_bottom),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.rectangle(
            frame,
            (bg_left, bg_top),
            (bg_right, bg_bottom),
            (60, 60, 60),
            thickness=1,
        )

        for i, (display, color, text) in enumerate(lines):
            y = h - self.margin - (n - 1 - i) * self.line_height
            label = f"[{display}] "
            cv2.putText(
                frame,
                label,
                (self.margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            # Measure label width so the text starts right after it.
            (lw, _lh), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                1,
            )
            cv2.putText(
                frame,
                text,
                (self.margin + lw, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        return frame
