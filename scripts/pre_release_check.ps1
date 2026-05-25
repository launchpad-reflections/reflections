# Pre-publish gate: TypeScript + Python checks and smoke imports.
# Run from repo root: .\scripts\pre_release_check.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==> Bun: install, typecheck, test"
bun install --frozen-lockfile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
bun run typecheck
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
bun test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Python: install, lint, format, test"
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python >= 3.10 required (same interpreter as 'python' on PATH)"
}
python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m ruff check packages tests scripts apps/viewer
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m black --check packages tests scripts apps/viewer
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Python: import smoke test"
python -c "import stream; import proactivity; import reflections; from proactivity.agent import ProactivityAgent"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Viewer: --help-only"
python -m apps.viewer --help-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Pre-release check passed."
