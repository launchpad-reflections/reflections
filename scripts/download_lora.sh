#!/usr/bin/env bash
# Prefetch the proactivity LoRA adapter from Hugging Face into the HF cache.
#
# Default model ID comes from reflections.config.LORA_MODEL_ID.
# Override: REFLECTIONS_LORA_MODEL_ID=my-org/my-adapter ./scripts/download_lora.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

HF_MODEL_ID="${REFLECTIONS_LORA_MODEL_ID:-$(
  "${PYTHON}" -c "from reflections.config import LORA_MODEL_ID; print(LORA_MODEL_ID)"
)}"

echo "Downloading LoRA adapter"
echo "  model: ${HF_MODEL_ID}"
echo "  cache: \${HF_HOME:-~/.cache/huggingface}/hub/"
echo "(Set REFLECTIONS_LORA_MODEL_ID to override the default HF repo.)"

# Prefer the modern `hf` CLI (huggingface_hub >= 0.34).
# Fall back to `huggingface-cli` for older installs.
if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli)
elif "${PYTHON}" -c "import huggingface_hub" >/dev/null 2>&1; then
  HF_CLI=("${PYTHON}" -m huggingface_hub.cli.huggingface_cli)
else
  echo "error: install huggingface_hub first (pip install -r requirements.txt)" >&2
  exit 1
fi

"${HF_CLI[@]}" download "${HF_MODEL_ID}"

echo "Done."
