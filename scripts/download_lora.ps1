# Prefetch the proactivity LoRA adapter from Hugging Face into the HF cache.
#
# Default model ID comes from reflections.config.LORA_MODEL_ID.
# Override: $env:REFLECTIONS_LORA_MODEL_ID = "my-org/my-adapter"

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($env:REFLECTIONS_LORA_MODEL_ID) {
    $HfModelId = $env:REFLECTIONS_LORA_MODEL_ID
} else {
    $HfModelId = python -c "from reflections.config import LORA_MODEL_ID; print(LORA_MODEL_ID)"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Downloading LoRA adapter"
Write-Host "  model: $HfModelId"
Write-Host "  cache: `$env:HF_HOME\hub\ (or ~\.cache\huggingface\hub\)"
Write-Host "(Set REFLECTIONS_LORA_MODEL_ID to override the default HF repo.)"

# Prefer the modern `hf` CLI (huggingface_hub >= 0.34).
# Fall back to `huggingface-cli` for older installs.
$hf = Get-Command hf -ErrorAction SilentlyContinue
$legacy = Get-Command huggingface-cli -ErrorAction SilentlyContinue

if ($hf) {
    & hf download $HfModelId
} elseif ($legacy) {
    & huggingface-cli download $HfModelId
} elseif (python -c "import huggingface_hub" 2>$null) {
    python -m huggingface_hub.cli.huggingface_cli download $HfModelId
} else {
    Write-Error "Install huggingface_hub first (pip install -r requirements.txt)"
}

Write-Host "Done."
