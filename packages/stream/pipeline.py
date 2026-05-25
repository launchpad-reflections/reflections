"""Pluggable frame/audio processing pipeline.

A Processor receives decoded video frames and/or audio chunks from the
StreamSource and can optionally overlay state onto frames during render.

Processors come in two flavors:
  - sync: on_video / on_audio run inline on the render thread. Use for
    cheap per-frame work (e.g. face detection).
  - async: on_video / on_audio enqueue work for a background thread. Use
    for slow models that must not block the video display. The worker
    thread updates shared state under a lock; draw() reads that state.
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


class Processor(ABC):
    mode: Literal["sync", "async"] = "sync"
    name: str = "processor"
    consumes: frozenset[str] = frozenset({"video", "audio"})

    def on_video(self, frame: np.ndarray, pts: float) -> None:
        pass

    def on_audio(self, samples: np.ndarray, sample_rate: int, pts: float) -> None:
        pass

    def draw(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def run_async(self, inbox: queue.Queue) -> None:
        """Override for async processors. Consume (kind, payload) tuples
        from `inbox` until a None sentinel is received."""
        while True:
            item = inbox.get()
            if item is None:
                return


class Pipeline:
    def __init__(self, processors: list[Processor]):
        self.processors = processors
        self._inboxes: dict[Processor, queue.Queue] = {}
        self._threads: list[threading.Thread] = []
        self._drop_counts: dict[Processor, int] = {}

    def start(self) -> None:
        for p in self.processors:
            if p.mode == "async":
                q: queue.Queue = queue.Queue(maxsize=256)
                self._inboxes[p] = q
                t = threading.Thread(
                    target=p.run_async, args=(q,), name=f"proc-{p.name}", daemon=True
                )
                t.start()
                self._threads.append(t)

    def stop(self, timeout: float = 1.0) -> None:
        for _p, q in self._inboxes.items():
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        self._inboxes.clear()

    def dispatch_video(self, frame: np.ndarray, pts: float) -> None:
        for p in self.processors:
            if "video" not in p.consumes:
                continue
            if p.mode == "sync":
                p.on_video(frame, pts)
            else:
                self._enqueue(p, ("video", frame, pts))

    def dispatch_audio(self, samples: np.ndarray, sample_rate: int, pts: float) -> None:
        for p in self.processors:
            if "audio" not in p.consumes:
                continue
            if p.mode == "sync":
                p.on_audio(samples, sample_rate, pts)
            else:
                self._enqueue(p, ("audio", samples, sample_rate, pts))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for p in self.processors:
            frame = p.draw(frame)
        return frame

    def _enqueue(self, p: Processor, item: tuple) -> None:
        q = self._inboxes.get(p)
        if q is None:
            return
        try:
            q.put_nowait(item)
        except queue.Full:
            # Drop oldest to keep latency bounded.
            try:
                q.get_nowait()
                q.put_nowait(item)
            except queue.Empty:
                pass
            n = self._drop_counts.get(p, 0) + 1
            self._drop_counts[p] = n
            if n == 1 or n % 200 == 0:
                logger.warning(
                    "[pipeline] %s inbox full; dropped %d items so far",
                    p.name,
                    n,
                )
