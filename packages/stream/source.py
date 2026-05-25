"""WHEP stream source: pulls WebRTC directly from MediaMTX, skipping
the RTSP re-serve path.

Runs an aiortc peer connection on a background asyncio thread and
bridges decoded video/audio frames into a thread-safe queue that the
synchronous `frames()` iterator drains. Reconnects on failure.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

logger = logging.getLogger(__name__)


@dataclass
class VideoItem:
    kind: str  # "video"
    frame: np.ndarray  # BGR24
    pts: float


@dataclass
class AudioItem:
    kind: str  # "audio"
    samples: np.ndarray  # float32, shape (n,) mono or (n, channels)
    sample_rate: int
    pts: float


_SENTINEL = object()


class StreamSource:
    def __init__(
        self,
        whep_url: str,
        reconnect_delay: float = 1.0,
        queue_size: int = 16,
    ):
        self.whep_url = whep_url
        self.reconnect_delay = reconnect_delay
        self._out: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stopped = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def close(self) -> None:
        self._stopped.set()
        try:
            self._out.put_nowait(_SENTINEL)
        except queue.Full:
            pass

    def frames(self) -> Iterator[VideoItem | AudioItem]:
        self._thread = threading.Thread(target=self._run_loop, name="whep-loop", daemon=True)
        self._thread.start()

        while not self._stopped.is_set():
            try:
                item = self._out.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            yield item

    # ---- asyncio side (runs on background thread) ----

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            logger.error("WHEP loop crashed: %s", e)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            try:
                self._out.put_nowait(_SENTINEL)
            except queue.Full:
                pass

    async def _main(self) -> None:
        while not self._stopped.is_set():
            pc = RTCPeerConnection()
            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")

            consumer_tasks: list[asyncio.Task] = []

            @pc.on("track")
            def on_track(track):
                consumer_tasks.append(asyncio.ensure_future(self._consume(track)))

            resource_url: str | None = None
            try:
                await pc.setLocalDescription(await pc.createOffer())
                await self._wait_ice_complete(pc)

                answer_sdp, resource_url = await self._whep_exchange(pc.localDescription.sdp)
                await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
                logger.info("WHEP connected.")

                await self._wait_disconnect(pc)
            except Exception as e:
                logger.warning("WHEP session error: %s", e)
            finally:
                for t in consumer_tasks:
                    t.cancel()
                for t in consumer_tasks:
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    await pc.close()
                except Exception:
                    pass
                if resource_url:
                    await self._whep_delete(resource_url)

            if self._stopped.is_set():
                return
            logger.info("Reconnecting WHEP in %ss...", self.reconnect_delay)
            await asyncio.sleep(self.reconnect_delay)

    async def _wait_ice_complete(self, pc: RTCPeerConnection) -> None:
        if pc.iceGatheringState == "complete":
            return
        done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def _on_change():
            if pc.iceGatheringState == "complete":
                done.set()

        await done.wait()

    async def _wait_disconnect(self, pc: RTCPeerConnection) -> None:
        done = asyncio.Event()

        @pc.on("connectionstatechange")
        def _on_change():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                done.set()

        await done.wait()

    async def _whep_exchange(self, offer_sdp: str) -> tuple[str, str | None]:
        timeout = aiohttp.ClientTimeout(total=10)
        async with (
            aiohttp.ClientSession(timeout=timeout) as http,
            http.post(
                self.whep_url,
                data=offer_sdp,
                headers={"Content-Type": "application/sdp"},
            ) as resp,
        ):
            body = await resp.text()
            if resp.status not in (200, 201):
                raise RuntimeError(f"WHEP POST failed: {resp.status} {body[:200]}")
            location = resp.headers.get("Location")
            resource_url = self._resolve_location(location) if location else None
            return body, resource_url

    async def _whep_delete(self, resource_url: str) -> None:
        try:
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.delete(resource_url) as resp:
                    await resp.read()
        except Exception:
            pass

    def _resolve_location(self, location: str) -> str:
        if location.startswith("http://") or location.startswith("https://"):
            return location
        from urllib.parse import urljoin

        return urljoin(self.whep_url, location)

    async def _consume(self, track) -> None:
        try:
            if track.kind == "video":
                while True:
                    frame = await track.recv()
                    pts = float(frame.pts * frame.time_base) if frame.pts else 0.0
                    img = frame.to_ndarray(format="bgr24")
                    self._publish(VideoItem(kind="video", frame=img, pts=pts))
            else:  # audio
                while True:
                    frame = await track.recv()
                    pts = float(frame.pts * frame.time_base) if frame.pts else 0.0
                    raw = frame.to_ndarray().astype(np.float32)
                    channels = (
                        getattr(frame.layout, "nb_channels", None)
                        or (len(frame.layout.channels) if frame.layout else 1)
                        or 1
                    )
                    is_planar = bool(getattr(frame.format, "is_planar", False))
                    # Normalize to shape (N, channels).
                    if channels <= 1:
                        samples = raw.reshape(-1)
                    elif is_planar:
                        # Planar formats: shape (channels, N) -> (N, channels)
                        samples = raw.reshape(channels, -1).T
                    else:
                        # Packed/interleaved: shape (1, N*channels) -> (N, channels)
                        samples = raw.reshape(-1, channels)
                    self._publish(
                        AudioItem(
                            kind="audio",
                            samples=samples,
                            sample_rate=frame.sample_rate,
                            pts=pts,
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("%s track ended: %s", track.kind, e)

    def _publish(self, item) -> None:
        try:
            self._out.put_nowait(item)
        except queue.Full:
            # Bound latency: drop oldest on backpressure.
            try:
                self._out.get_nowait()
                self._out.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass
