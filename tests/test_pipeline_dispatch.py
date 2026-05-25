"""Tests for stream.pipeline dispatch behavior."""

from __future__ import annotations

import queue

import numpy as np
from stream.pipeline import Pipeline, Processor


class SyncVideo(Processor):
    mode = "sync"
    name = "sync-video"
    consumes = frozenset({"video"})

    def __init__(self) -> None:
        self.frames: list[tuple[np.ndarray, float]] = []

    def on_video(self, frame: np.ndarray, pts: float) -> None:
        self.frames.append((frame, pts))


class AsyncAudio(Processor):
    mode = "async"
    name = "async-audio"
    consumes = frozenset({"audio"})

    def __init__(self) -> None:
        self.chunks: list[tuple[np.ndarray, int, float]] = []

    def run_async(self, inbox: queue.Queue) -> None:
        while True:
            item = inbox.get()
            if item is None:
                return
            kind, samples, sample_rate, pts = item
            assert kind == "audio"
            self.chunks.append((samples, sample_rate, pts))


class AudioOnlySync(Processor):
    mode = "sync"
    name = "audio-only"
    consumes = frozenset({"audio"})

    def __init__(self) -> None:
        self.chunks: list[int] = []

    def on_audio(self, samples: np.ndarray, sample_rate: int, pts: float) -> None:
        self.chunks.append(sample_rate)


def test_dispatch_routes_by_consumes() -> None:
    sync = SyncVideo()
    audio = AudioOnlySync()
    pipeline = Pipeline([sync, audio])
    pipeline.start()

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    samples = np.zeros(160, dtype=np.int16)
    pipeline.dispatch_video(frame, 1.0)
    pipeline.dispatch_audio(samples, 16000, 2.0)

    assert len(sync.frames) == 1
    assert audio.chunks == [16000]
    pipeline.stop()


def test_async_audio_reaches_worker() -> None:
    proc = AsyncAudio()
    pipeline = Pipeline([proc])
    pipeline.start()

    samples = np.ones(80, dtype=np.int16)
    pipeline.dispatch_audio(samples, 48000, 3.5)
    pipeline.stop(timeout=2.0)

    assert len(proc.chunks) == 1
    assert proc.chunks[0][1] == 48000
