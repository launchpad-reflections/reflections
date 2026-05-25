"""LR-ASD active speaker detection model loader (CPU inference only).

Vendored from https://github.com/Junhua-Liao/LR-ASD. We keep only the
network modules needed for inference and add a Linear(128, 2) head that
matches the `lossAV.FC` layer from the original checkpoint.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import ASD_Model

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = Path(__file__).parent / "weights" / "finetuning_TalkSet.model"


class ASDInference(nn.Module):
    """Wraps ASD_Model + the lossAV FC head for a single forward call."""

    def __init__(self):
        super().__init__()
        self.model = ASD_Model()
        self.fc = nn.Linear(128, 2)

    def forward(self, audio_mfcc: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """
        audio_mfcc: (B, T_audio, 13) float32
        visual:     (B, T_video, 112, 112) float32 in 0..255 range
        returns per-frame speaking logits: shape (B, T_video)
        """
        outsAV, _ = self.model(audio_mfcc, visual)  # (B*T_video, 128)
        logits = self.fc(outsAV)                    # (B*T_video, 2)
        speak_logit = logits[:, 1] - logits[:, 0]   # log-odds of speaking
        return speak_logit.view(visual.shape[0], visual.shape[1])


_singleton: ASDInference | None = None


def load(weights_path: str | os.PathLike | None = None) -> ASDInference:
    """Build the model, load weights, return an eval-mode instance.
    Idempotent — subsequent calls return the same cached model."""
    global _singleton
    if _singleton is not None:
        return _singleton

    # CPU thread hygiene. Leave one core for aiortc/OpenCV.
    try:
        n = max(1, (os.cpu_count() or 2) // 2)
        torch.set_num_threads(n)
    except Exception:
        pass
    torch.set_grad_enabled(False)

    path = Path(weights_path) if weights_path else _WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"LR-ASD weights not found at {path}")

    raw = torch.load(path, map_location="cpu")
    # Checkpoint is the full ASD() state_dict: model.* + lossAV.FC.* + lossV.FC.*
    # We need model.* (renamed) and lossAV.FC.* (→ fc.*).
    state: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            state[k] = v  # "model.visualEncoder..." lines up with ASDInference.model
        elif k.startswith("lossAV.FC."):
            state["fc." + k[len("lossAV.FC."):]] = v
        # lossV.FC.* is discarded — we don't use the visual-only head.

    net = ASDInference()
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        logger.warning(
            "[lr_asd] missing keys: %s%s",
            missing[:5],
            "..." if len(missing) > 5 else "",
        )
    if unexpected:
        logger.warning(
            "[lr_asd] unexpected keys: %s%s",
            unexpected[:5],
            "..." if len(unexpected) > 5 else "",
        )
    net.eval()

    _singleton = net
    return net


@torch.inference_mode()
def predict(video_batch: np.ndarray, mfcc_batch: np.ndarray) -> np.ndarray:
    """
    video_batch: (B, T_video, 112, 112) uint8 or float32, 0..255
    mfcc_batch:  (B, T_audio, 13)       float32
    returns:     (B, T_video) float32 speaking log-odds
    """
    net = load()
    v = torch.from_numpy(np.ascontiguousarray(video_batch)).float()
    a = torch.from_numpy(np.ascontiguousarray(mfcc_batch)).float()
    logits = net(a, v)
    return logits.cpu().numpy()
