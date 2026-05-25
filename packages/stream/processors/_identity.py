"""Face-embedding-based identity resolution for ASD tracks.

Uses InsightFace's MobileFaceNet (w600k_mbf.onnx from the buffalo_s
pack) via onnxruntime for face embeddings. This is the standard
CPU-realtime ArcFace variant: 512-d output, ~5-10 ms per face on
modern CPUs, ~13 MB weights. We align with a 5-point similarity
transform from YuNet landmarks to the ArcFace template, then run the
ONNX forward pass.

A gallery of embeddings per named identity is stored on disk as a .npz
(one array per identity, shape (K, 512)) with a JSON sidecar.

When ASD decides a track is speaking, this resolver is called with the
full BGR frame and the YuNet face row. It aligns + embeds the face and
either matches against an existing identity or mints a new 'Person N'.
The tid → name decision is returned and the gallery is updated in
memory. On flush() we optionally cluster session-new identities to
merge ones that turn out to be the same person seen in multiple
separate tracks.

The gallery is only touched by the ASD inference worker thread; all
file I/O happens from there.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard ArcFace 112x112 5-point reference template. Index order
# matches YuNet's landmark order (right_eye, left_eye, nose,
# right_mouth, left_mouth in the image frame) so we can pass YuNet
# points straight through to estimateAffinePartial2D.
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-8 else (v / n)


def _align_face(frame_bgr: np.ndarray, landmarks_5x2: np.ndarray) -> np.ndarray | None:
    """Similarity-transform the face to a 112x112 ArcFace crop."""
    if landmarks_5x2.shape != (5, 2):
        return None
    M, _ = cv2.estimateAffinePartial2D(
        landmarks_5x2.astype(np.float32),
        _ARCFACE_TEMPLATE,
        method=cv2.LMEDS,
    )
    if M is None:
        return None
    return cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0.0)


class IdentityResolver:
    def __init__(
        self,
        model_path: Path,
        gallery_path: Path,
        index_path: Path,
        *,
        match_threshold: float = 0.42,
        margin: float = 0.08,
        strong_match_threshold: float = 0.45,
        max_per_identity: int = 10,
        add_cooldown_sec: float = 2.0,
        merge_threshold: float = 0.45,
        debug: bool = False,
    ):
        self.match_threshold = match_threshold
        self.margin = margin
        self.strong_match_threshold = strong_match_threshold
        self.max_per_identity = max_per_identity
        self.add_cooldown_sec = add_cooldown_sec
        self.merge_threshold = merge_threshold
        self.debug = debug

        self.gallery_path = Path(gallery_path)
        self.index_path = Path(index_path)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Face recognition model missing: {model_path}\n"
                "Download w600k_mbf.onnx (MobileFaceNet, ArcFace, ~13 MB):\n"
                "  1. Grab the buffalo_s model pack from InsightFace:\n"
                "     https://github.com/deepinsight/insightface/releases/"
                "download/v0.7/buffalo_s.zip\n"
                "  2. Unzip and copy w600k_mbf.onnx to:\n"
                f"     {model_path}"
            )

        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        # buffalo_s mbf is 512-d.
        self._dim = int(self._sess.get_outputs()[0].shape[-1])

        self._lock = threading.Lock()
        # name -> ndarray (K, D) of L2-normalized embeddings
        self._gallery: dict[str, np.ndarray] = {}
        # Newly-minted "Person N" names sit here until something
        # confirms they correspond to a real audible speaker (the
        # Soniox fusion layer calls promote_pending). This keeps the
        # persistent gallery from filling up with phantom Person N
        # entries that were created off a single ASD positive on a
        # face that turned out to be silent / a YuNet false positive.
        self._pending: dict[str, np.ndarray] = {}
        # name -> {"created": float, "count": int}
        self._meta: dict[str, dict] = {}
        self._next_auto_id: int = 1
        self._session_new: set[str] = set()
        self._session_touched: set[str] = set()
        self._last_add_ts: dict[int, float] = {}

        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        try:
            if self.index_path.exists():
                with open(self.index_path, encoding="utf-8") as f:
                    idx = json.load(f)
                self._next_auto_id = int(idx.get("next_auto_id", 1))
                for entry in idx.get("identities", []):
                    name = entry.get("name")
                    if name:
                        self._meta[name] = {
                            "created": float(entry.get("created", time.time())),
                            "count": int(entry.get("count", 0)),
                        }
            if self.gallery_path.exists():
                with np.load(self.gallery_path, allow_pickle=False) as npz:
                    for name in npz.files:
                        arr = np.asarray(npz[name], dtype=np.float32)
                        # Guard against dim drift between model versions.
                        if arr.ndim == 2 and arr.shape[1] == self._dim:
                            self._gallery[name] = arr
                            self._meta.setdefault(
                                name,
                                {"created": time.time(), "count": int(arr.shape[0])},
                            )
                        elif arr.ndim == 2:
                            logger.warning(
                                "[identity] dropping '%s' from gallery: " "dim %d != model dim %d",
                                name,
                                arr.shape[1],
                                self._dim,
                            )
            if self.debug:
                logger.debug(
                    "[identity] loaded gallery: " "%d identities, %d embeddings",
                    len(self._gallery),
                    sum(a.shape[0] for a in self._gallery.values()),
                )
        except Exception as e:
            logger.warning("[identity] gallery load failed: %s; starting fresh", e)
            self._gallery = {}
            self._meta = {}
            self._next_auto_id = 1

    def rename_identity(self, old_name: str, new_name: str) -> bool:
        """Rename an identity in the gallery before flush. Returns True if found."""
        with self._lock:
            if old_name not in self._gallery:
                return False
            if new_name in self._gallery:
                combined = np.vstack([self._gallery[new_name], self._gallery[old_name]])
                if combined.shape[0] > self.max_per_identity:
                    combined = combined[-self.max_per_identity :]
                self._gallery[new_name] = combined
                del self._gallery[old_name]
                self._meta.pop(old_name, None)
                self._meta.setdefault(new_name, {"created": time.time(), "count": 0})
                self._meta[new_name]["count"] = int(combined.shape[0])
            else:
                self._gallery[new_name] = self._gallery.pop(old_name)
                meta = self._meta.pop(old_name, {"created": time.time(), "count": 0})
                self._meta[new_name] = meta
            self._session_new.discard(old_name)
            self._session_touched.discard(old_name)
            self._session_touched.add(new_name)
            return True

    def flush(self) -> None:
        with self._lock:
            merges = self._cluster_merge_locked()
            self._write_atomic_locked()
        if merges and self.debug:
            for src, dst in merges:
                logger.debug("[identity] merged '%s' -> '%s' on flush", src, dst)

    def _write_atomic_locked(self) -> None:
        try:
            self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
            # np.savez appends .npz to paths that don't already end in it,
            # so use a tmp path that already ends in .npz.
            tmp_npz = self.gallery_path.with_name(self.gallery_path.stem + "_tmp.npz")
            np.savez(tmp_npz, **self._gallery)
            os.replace(tmp_npz, self.gallery_path)

            index = {
                "version": 1,
                "model_dim": self._dim,
                "next_auto_id": self._next_auto_id,
                "identities": [
                    {
                        "name": name,
                        "count": int(self._gallery[name].shape[0]),
                        "created": float(self._meta.get(name, {}).get("created", time.time())),
                    }
                    for name in self._gallery
                ],
            }
            tmp_json = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            os.replace(tmp_json, self.index_path)
        except Exception as e:
            logger.error("[identity] gallery flush failed: %s", e)

    # ---- embedding ----

    def _embed(
        self,
        frame_bgr: np.ndarray,
        face_row: np.ndarray,
    ) -> np.ndarray | None:
        # YuNet row: [x, y, w, h, lm0x, lm0y, ..., lm4x, lm4y, score]
        row = np.asarray(face_row, dtype=np.float32).reshape(-1)
        if row.size < 14:
            return None
        lms = row[4:14].reshape(5, 2)
        aligned = _align_face(frame_bgr, lms)
        if aligned is None:
            return None
        # BGR -> RGB, [-1, 1], NCHW.
        img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))[None, ...]
        try:
            out = self._sess.run(None, {self._input_name: img})[0]
        except Exception as e:
            if self.debug:
                logger.debug("[identity] onnx forward failed: %s", e)
            return None
        emb = np.asarray(out, dtype=np.float32).reshape(-1)
        return _l2(emb)

    @staticmethod
    def _top_k_mean(sims: np.ndarray, k: int = 3) -> float:
        if sims.size == 0:
            return -1.0
        k = min(k, sims.size)
        idx = np.argpartition(-sims, k - 1)[:k]
        return float(np.mean(sims[idx]))

    def _best_matches_locked(self, emb: np.ndarray) -> list[tuple[str, float]]:
        """Cosine matches against BOTH gallery and pending entries.
        Pending faces are still real observations from this session, so
        matching against them prevents the same face from being minted
        as Person 4, then Person 5, then Person 6 while it's stuck in
        pending limbo waiting for Soniox confirmation."""
        out: list[tuple[str, float]] = []
        for name, arr in self._gallery.items():
            out.append((name, self._top_k_mean(arr @ emb, 3)))
        for name, arr in self._pending.items():
            if name in self._gallery:
                continue  # already counted above
            out.append((name, self._top_k_mean(arr @ emb, 3)))
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out

    def _append_locked(self, name: str, emb: np.ndarray) -> None:
        arr = self._gallery.get(name)
        if arr is None:
            self._gallery[name] = emb.reshape(1, -1).astype(np.float32)
            self._meta[name] = {"created": time.time(), "count": 1}
        else:
            new = np.vstack([arr, emb.reshape(1, -1).astype(np.float32)])
            if new.shape[0] > self.max_per_identity:
                new = new[-self.max_per_identity :]
            self._gallery[name] = new
            self._meta.setdefault(name, {"created": time.time(), "count": 0})
            self._meta[name]["count"] = int(new.shape[0])
        self._session_touched.add(name)

    def _append_pending_locked(self, name: str, emb: np.ndarray) -> None:
        """Same as _append_locked but writes to the pending dict, which
        is NOT persisted to disk on flush. Pending entries stay alive
        until promote_pending() moves them into the gallery."""
        arr = self._pending.get(name)
        row = emb.reshape(1, -1).astype(np.float32)
        if arr is None:
            self._pending[name] = row
        else:
            new = np.vstack([arr, row])
            if new.shape[0] > self.max_per_identity:
                new = new[-self.max_per_identity :]
            self._pending[name] = new

    def promote_pending(self, name: str) -> bool:
        """Move a pending identity's embeddings into the permanent
        gallery. Called by ASDProcessor.confirm_identity_for_name once
        the Soniox fusion layer has locked a Soniox speaker id to this
        name (i.e. we now have audio confirmation that this face has
        actually been speaking). Returns True if a promotion happened."""
        with self._lock:
            arr = self._pending.pop(name, None)
            if arr is None or arr.size == 0:
                return False
            existing = self._gallery.get(name)
            if existing is None:
                self._gallery[name] = arr
                self._meta[name] = {"created": time.time(), "count": int(arr.shape[0])}
            else:
                combined = np.vstack([existing, arr])
                if combined.shape[0] > self.max_per_identity:
                    combined = combined[-self.max_per_identity :]
                self._gallery[name] = combined
                self._meta.setdefault(name, {"created": time.time(), "count": 0})
                self._meta[name]["count"] = int(combined.shape[0])
            self._session_new.add(name)
            self._session_touched.add(name)
        if self.debug:
            logger.debug("[identity] promoted pending '%s' -> gallery", name)
        return True

    def resolve(
        self,
        frame_bgr: np.ndarray,
        face_row: np.ndarray,
        tid: int,
    ) -> str | None:
        emb = self._embed(frame_bgr, face_row)
        if emb is None:
            return None

        now = time.monotonic()
        with self._lock:
            matches = self._best_matches_locked(emb)
            best = matches[0] if matches else None
            second = matches[1][1] if len(matches) >= 2 else -1.0

            if (
                best is not None
                and best[1] >= self.match_threshold
                and (best[1] - second) >= self.margin
            ):
                name = best[0]
                cool_ok = (now - self._last_add_ts.get(tid, 0.0)) >= self.add_cooldown_sec
                if best[1] >= self.strong_match_threshold and cool_ok:
                    # Strong match: append to wherever this name currently
                    # lives. Pending names stay pending until they're
                    # confirmed; gallery names just get a new embedding.
                    if name in self._gallery:
                        self._append_locked(name, emb)
                    else:
                        self._append_pending_locked(name, emb)
                    self._last_add_ts[tid] = now
                where = "gallery" if name in self._gallery else "pending"
                logger.info(
                    "[identity] Track %d matched to %s from %s! (sim=%.3f)",
                    tid,
                    name,
                    where,
                    best[1],
                )
                return name

            # No confident match: mint a new auto identity. Stash it in
            # the pending slot — it will be promoted into the gallery
            # only once the Soniox fusion layer confirms this face was
            # actually speaking (via promote_pending).
            name = f"Person {self._next_auto_id}"
            while name in self._gallery or name in self._pending:
                self._next_auto_id += 1
                name = f"Person {self._next_auto_id}"
            self._next_auto_id += 1
            self._append_pending_locked(name, emb)
            self._last_add_ts[tid] = now
            best_str = f"closest={best[0]}@{best[1]:.3f}" if best is not None else "gallery empty"
            logger.info(
                "[identity] Track %d is new → minted %s (pending; %s)",
                tid,
                name,
                best_str,
            )
            return name

    # ---- end-of-session merge clustering ----

    def _cluster_merge_locked(self) -> list[tuple[str, str]]:
        if not self._session_new:
            return []

        merges: list[tuple[str, str]] = []

        def centroid(name: str) -> np.ndarray | None:
            arr = self._gallery.get(name)
            if arr is None or arr.shape[0] == 0:
                return None
            return _l2(arr.mean(axis=0))

        def _auto_num(n: str) -> int:
            try:
                return int(n.split()[-1])
            except Exception:
                return 10**9

        changed = True
        while changed:
            changed = False
            for src in list(self._session_new):
                if src not in self._gallery:
                    self._session_new.discard(src)
                    continue
                src_c = centroid(src)
                if src_c is None:
                    continue

                best_name: str | None = None
                best_sim: float = -1.0
                for other, arr in self._gallery.items():
                    if other == src:
                        continue
                    s = self._top_k_mean(arr @ src_c, 3)
                    if s > best_sim:
                        best_sim, best_name = s, other

                if best_name is None or best_sim < self.merge_threshold:
                    continue

                dst = best_name
                # Prefer non-session-new (named or older) as destination.
                if dst in self._session_new and _auto_num(dst) > _auto_num(src):
                    dst, src = src, dst

                src_arr = self._gallery.pop(src)
                dst_arr = self._gallery.get(dst, np.zeros((0, src_arr.shape[1]), dtype=np.float32))
                combined = np.vstack([dst_arr, src_arr])
                if combined.shape[0] > self.max_per_identity:
                    combined = combined[-self.max_per_identity :]
                self._gallery[dst] = combined
                self._meta.pop(src, None)
                self._meta.setdefault(dst, {"created": time.time(), "count": 0})
                self._meta[dst]["count"] = int(combined.shape[0])
                self._session_new.discard(src)
                self._session_touched.discard(src)
                self._session_touched.add(dst)
                merges.append((src, dst))
                changed = True
                break

        return merges
