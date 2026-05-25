# Model weights

Reflections uses several model files. Some ship with the repo, some auto-download on first run, and some must be fetched manually.

**Python ≥ 3.10** is required for the proactivity classifier stack (`peft==0.19.x`).

## Summary

| Model | Purpose | Location | How to obtain |
| --- | --- | --- | --- |
| YuNet | Face detection (ASD pipeline) | `models/face_detection_yunet_2023mar.onnx` | Manual download |
| MobileFaceNet (w600k_mbf) | Face embeddings / identity | `models/w600k_mbf.onnx` | Manual download |
| Qwen3 1.7B | Proactivity classifier base | Hugging Face cache (`Qwen/Qwen3-1.7B`) | Auto-download (~3.4 GB) |
| LoRA adapter | Actionability gate fine-tune | Hugging Face Hub (`REFLECTIONS_LORA_MODEL_ID`) | Auto-download or prefetch via scripts |
| LR-ASD | Active speaker detection | `packages/third_party/lr_asd/weights/` | Ships with repo |
| Face gallery | Persistent identity store | `models/face_gallery.npz`, `models/face_gallery.json` | Created at runtime |

---

## ONNX models (manual download)

Place both files in the repo-root `models/` directory (create it if missing).

### YuNet — face detection

- **File:** `models/face_detection_yunet_2023mar.onnx`
- **Source:** [OpenCV Zoo — face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Used by:** `packages/stream/processors/asd/` (YuNet detector + IoU tracker)

```bash
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

### MobileFaceNet (w600k_mbf) — face recognition

- **File:** `models/w600k_mbf.onnx`
- **Source:** [`buffalo_s.zip`](https://github.com/deepinsight/insightface/releases/tag/v0.7) (InsightFace v0.7 pack)
- **Used by:** `packages/stream/processors/_identity.py` (512-d ArcFace embeddings for the face gallery)

```bash
curl -L -o buffalo_s.zip \
  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
unzip buffalo_s.zip w600k_mbf.onnx
mv w600k_mbf.onnx models/
rm buffalo_s.zip
```

**Windows (PowerShell):** download and extract with your preferred archive tool, then copy `w600k_mbf.onnx` to `models\`.

---

## Qwen3 1.7B base (auto-download)

- **Hugging Face ID:** `Qwen/Qwen3-1.7B`
- **Size:** ~3.4 GB (weights + tokenizer)
- **Used by:** `packages/proactivity/classifier.py` (`load_model()`)
- **Cache:** `$HF_HOME/hub/` or `~/.cache/huggingface/hub/` (override with `HF_HOME`)

The base model downloads automatically the first time the classifier loads (viewer startup, `scripts/smoke_server.py`, etc.). No manual step is required if you have network access and sufficient disk space.

To prefetch without running the full app:

```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')"
```

---

## LoRA adapter (Hugging Face Hub)

- **HF model ID:** `rushilsaraf/qwen3-actionable-v2-adapter` (override with `REFLECTIONS_LORA_MODEL_ID`)
- **Config:** `packages/reflections/config.py` → `LORA_MODEL_ID`
- **PEFT version:** 0.19.1
- **Used by:** `packages/proactivity/classifier.py` (`PeftModel.from_pretrained(base, LORA_MODEL_ID)`)
- **Cache:** same Hugging Face hub cache as the Qwen base (`$HF_HOME/hub/`)

The classifier loads the adapter from Hugging Face Hub at runtime. If the Hub download fails (offline dev), it falls back to a local directory at `packages/proactivity/qwen3-actionable-v2-adapter/` when present.

### Prefetch from Hugging Face

```bash
# Linux / macOS
./scripts/download_lora.sh

# Windows
.\scripts\download_lora.ps1
```

Requires `huggingface_hub` (installed transitively via `transformers`). Authenticate for private repos:

```bash
huggingface-cli login
```

Expected adapter files in the HF repo include `adapter_config.json`, `adapter_model.safetensors`, and tokenizer assets.

### Model card

LoRA adapter for **Qwen3-1.7B** that classifies whether the latest sentence in a smart-glasses conversation transcript warrants **proactive assistant intervention** (label `1`) or should be ignored (label `0`).

Reflections uses this model as a fast, on-device gate (~200 ms on Apple Silicon with MPS). When P(actionable) exceeds `GLASSES_GATE_THRESHOLD` (default 0.25), the pipeline escalates to Claude for reasoning and optional TTS output.

**Architecture:** Qwen3-1.7B decoder-only transformer with rank-8 LoRA adapters on attention and MLP projections. Binary score = softmax over vocabulary logits for tokens `0` and `1` at the `<label>` position.

**Training:** LoRA SFT with Unsloth-optimized Qwen3-1.7B 4-bit base (`unsloth/qwen3-1.7b-unsloth-bnb-4bit`); runtime merges onto **`Qwen/Qwen3-1.7B`** in float16.

| Parameter | Value |
|---|---|
| PEFT type | LoRA |
| Rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Task type | CAUSAL_LM |
| PEFT version | 0.19.1 |

**Gate thresholds (distinct knobs):**

- `GLASSES_GATE_THRESHOLD` (default **0.25**) — live glasses path in `agent/worker.py`
- `REASONING_TRIGGER` (**0.45**) — smoke/test tooling only; enables reasoning generation

**Out of scope:** general chat, non-English without retraining, medical/legal/safety-critical decisions, standalone deployment without the Reflections context pipeline.

**License:** Apache 2.0 for Qwen3 base weights; adapter checkpoint distributed under the Reflections repository MIT license. Combined use remains subject to the [Qwen3 license](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE).

**Hub page:** https://huggingface.co/rushilsaraf/qwen3-actionable-v2-adapter

---

## Runtime-generated artifacts

These are **not** checked in and are excluded from Docker builds (see `.dockerignore`):

| File | Purpose |
| --- | --- |
| `models/face_gallery.npz` | Face embedding arrays per identity |
| `models/face_gallery.json` | Gallery metadata (names, counts, timestamps) |

Delete both files to reset the face identity gallery. See [PRIVACY.md](PRIVACY.md).

---

## LR-ASD weights (bundled)

Active speaker detection weights live under `packages/third_party/lr_asd/weights/` and are referenced by `packages/stream/processors/asd/`. No download step is required.

## Related docs

- [SETUP.md](SETUP.md) — when to download weights in the install flow
- [CONFIGURATION.md](CONFIGURATION.md) — `HF_HOME`, `REFLECTIONS_LORA_MODEL_ID`
