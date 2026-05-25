"""YuNet face detection."""

from __future__ import annotations

import cv2
import numpy as np
from reflections.config import REPO_ROOT

YUNET_PATH = REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    def __init__(self) -> None:
        self._detector: cv2.FaceDetectorYN | None = None
        self._det_input_size: tuple[int, int] = (0, 0)

    def detect(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        h, w = frame.shape[:2]
        det = self._ensure_detector(w, h)
        _, faces = det.detect(frame)
        out: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh = f[0], f[1], f[2], f[3]
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(w, x + bw))
            y2 = int(min(h, y + bh))
            if x2 > x1 and y2 > y1:
                # SFace alignCrop wants the full 15-element YuNet row.
                out.append(((x1, y1, x2, y2), f.reshape(1, -1).astype(np.float32)))
        return out

    def _ensure_detector(self, w: int, h: int) -> cv2.FaceDetectorYN:
        if self._detector is None:
            if not YUNET_PATH.exists():
                raise FileNotFoundError(f"YuNet weights missing: {YUNET_PATH}")
            self._detector = cv2.FaceDetectorYN.create(
                str(YUNET_PATH),
                config="",
                input_size=(w, h),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5,
            )
            self._det_input_size = (w, h)
        elif (w, h) != self._det_input_size:
            self._detector.setInputSize((w, h))
            self._det_input_size = (w, h)
        return self._detector
