# Third-Party Notices

Reflections incorporates or depends on the following third-party software and
models. This file summarizes attribution and license terms. See also NOTICE
and LICENSE in the repository root.

---

## MentraOS Camera Example App (template)

- **Source:** https://github.com/Mentra-Community/MentraOS-Camera-Example-App
- **Use in Reflections:** Initial Bun/TypeScript app-server scaffold, MentraOS
  SDK integration, and camera-session patterns.
- **License:** MIT (Mentra Community)

---

## LR-ASD (Lightweight and Robust Active Speaker Detection)

- **Source:** https://github.com/Junhua-Liao/LR-ASD
- **Paper:** Liao et al., *LR-ASD: Lightweight and Robust Network for Active
  Speaker Detection*, IJCV 2025
- **Use in Reflections:** Vendored under `packages/third_party/lr_asd/` for
  live audio-visual active speaker detection.
- **License:** MIT — see `packages/third_party/lr_asd/LICENSE`

---

## InsightFace MobileFaceNet (ArcFace, w600k_mbf)

- **Source:** https://github.com/deepinsight/insightface (buffalo_s model pack)
- **Use in Reflections:** Face embedding and cross-session identity matching
  (`models/w600k_mbf.onnx`, downloaded separately).
- **License:** InsightFace model weights are subject to the InsightFace project
  terms; the buffalo_s pack is widely distributed for non-commercial and
  research use. Review the upstream release notes before redistribution.

---

## OpenCV YuNet (face detection)

- **Source:** https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- **Use in Reflections:** Face detection for ASD crops and identity alignment
  (`models/face_detection_yunet_2023mar.onnx`, downloaded separately).
- **License:** Apache License 2.0 (OpenCV Zoo models)

---

## MediaMTX

- **Source:** https://github.com/bluenviron/mediamtx
- **Use in Reflections:** WHIP/WHEP media server for glasses → laptop streaming
  (binary drop-in under `mediamtx/`, not committed).
- **License:** MIT — see `mediamtx/LICENSE` when the binary is present locally

---

## Qwen3 (base model and LoRA adapter)

- **Base model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Training base (adapter checkpoint):**
  [unsloth/qwen3-1.7b-unsloth-bnb-4bit](https://huggingface.co/unsloth/qwen3-1.7b-unsloth-bnb-4bit)
- **Adapter:** [`rushilsaraf/qwen3-actionable-v2-adapter`](https://huggingface.co/rushilsaraf/qwen3-actionable-v2-adapter)
  on Hugging Face Hub (Reflections LoRA weights, trained for binary
  actionability classification). Hub ID is configurable via
  `REFLECTIONS_LORA_MODEL_ID`; see `docs/MODELS.md`.
- **Use in Reflections:** On-device gate that scores whether a transcript turn
  warrants proactive assistant intervention (~200 ms inference).
- **License:** Apache License 2.0 (Qwen3 open-weight models). The Reflections
  LoRA adapter is distributed under the same terms as this repository (MIT);
  downstream use of Qwen3 weights remains subject to the Apache 2.0 license.

---

## Anthropic API (Claude)

- **Service:** https://www.anthropic.com/
- **Use in Reflections:** Memory distillation, name resolution, and proactivity
  reasoning when the local classifier gate passes.
- **Terms:** Use is governed by the [Anthropic Terms of Service](https://www.anthropic.com/legal/terms)
  and [Commercial Terms](https://www.anthropic.com/legal/commercial-terms)
  as applicable. API keys are required; no Anthropic model weights are
  redistributed in this repository.

---

## Soniox (speech-to-text)

- **Service:** https://soniox.com/
- **Use in Reflections:** Real-time streaming transcription over WebSocket.
- **Terms:** Use is governed by the [Soniox Terms of Service](https://soniox.com/terms).
  API keys are required; no Soniox model weights are redistributed in this
  repository.

---

## Other runtime dependencies

Python and JavaScript dependencies (PyTorch, Transformers, PEFT, aiortc,
`@mentra/sdk`, etc.) retain their respective upstream licenses. Refer to
`requirements.txt`, `package.json`, and lockfiles for the full dependency tree.
