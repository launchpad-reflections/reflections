# Fresh-clone sanity check — no glasses, MediaMTX, or ngrok required.
# Run from anywhere: .\scripts\verify_fresh_setup.ps1

$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Results = [ordered]@{}
$Failed = 0

function Test-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Name"
    try {
        & $Action
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            throw "exit code $LASTEXITCODE"
        }
        $script:Results[$Name] = "PASS"
        Write-Host "    PASS"
    } catch {
        $script:Results[$Name] = "FAIL"
        $script:Failed++
        Write-Host "    FAIL: $_" -ForegroundColor Red
    }
}

Write-Host "Reflections fresh-clone verify"
Write-Host "  repo: $RepoRoot"

Test-Step "Python >= 3.10" {
    python -c "import sys; v = sys.version_info; print(f'    {sys.executable} ({sys.version.split()[0]})'); raise SystemExit(0 if v >= (3, 10) else 1)"
}

Test-Step 'pip install -e ".[dev]"' {
    python -m pip install -e ".[dev]"
}

Test-Step "pytest -q" {
    python -m pytest -q
}

Test-Step "bun install --frozen-lockfile" {
    bun install --frozen-lockfile
}

Test-Step "bun run typecheck" {
    bun run typecheck
}

Test-Step "bun test" {
    bun test
}

Test-Step "python -m apps.viewer --help-only" {
    python -m apps.viewer --help-only
}

Test-Step "import stream, proactivity, reflections, ProactivityAgent" {
    python -c @"
import stream
import proactivity
import reflections
from proactivity.agent import ProactivityAgent
print('imports ok')
"@
}

Write-Host ""
Write-Host "========== Summary =========="
foreach ($entry in $Results.GetEnumerator()) {
    $color = if ($entry.Value -eq "PASS") { "Green" } else { "Red" }
    Write-Host ("  [{0}] {1}" -f $entry.Value, $entry.Key) -ForegroundColor $color
}
Write-Host "============================="

if ($Failed -gt 0) {
    Write-Host ""
    Write-Host "$Failed check(s) failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All checks passed." -ForegroundColor Green
exit 0
