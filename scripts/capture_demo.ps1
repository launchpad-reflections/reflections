# Demo asset capture helper (documentation only — no live stream required).
#
# Prints ffmpeg / GIF steps for docs/demo.gif and docs/dashboard.png.
# Optionally opens the proactivity dashboard in your browser.
#
# Usage:
#   .\scripts\capture_demo.ps1              # print instructions + open dashboard
#   .\scripts\capture_demo.ps1 -NoBrowser   # print instructions only

param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$DashboardPort = if ($env:DASHBOARD_PORT) { $env:DASHBOARD_PORT } else {
    python -c "from reflections.config import DASHBOARD_PORT; print(DASHBOARD_PORT)"
    if ($LASTEXITCODE -ne 0) { "8766" }
}
$DashboardUrl = "http://127.0.0.1:$DashboardPort/"

Write-Host ""
Write-Host "=== Reflections demo asset capture (Windows) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Full guide: docs/DEMO_ASSETS.md"
Write-Host ""
Write-Host "WARNING: Do NOT use files from recordings/ - they may contain real audio."
Write-Host "         Use synthetic or fictional speech only."
Write-Host ""
Write-Host '--- Prerequisites (live session) ---'
Write-Host "  1. ngrok http 3000"
Write-Host "  2. .\mediamtx\mediamtx.exe .\mediamtx\mediamtx.yml"
Write-Host "  3. bun run dev"
Write-Host '  4. python -m apps.viewer          (Mentra Live window)'
Write-Host "  5. python -m proactivity.dashboard"
Write-Host ""
Write-Host '--- docs/demo.gif (~10 s viewer + proactive TTS) ---'
Write-Host '  Record the Mentra Live OpenCV window (Win+G Game Bar or ffmpeg gdigrab).'
Write-Host ""
Write-Host '  ffmpeg example (adjust offset/size to your window):'
Write-Host '    ffmpeg -f gdigrab -framerate 30 -offset_x 100 -offset_y 100 -video_size 1280x720 -t 10 -i desktop docs/demo_raw.mp4'
Write-Host ""
Write-Host "  Convert to GIF:"
Write-Host '    ffmpeg -i docs/demo_raw.mp4 -vf "fps=15,scale=960:-1:flags=lanczos,palettegen" docs/demo_palette.png'
Write-Host '    ffmpeg -i docs/demo_raw.mp4 -i docs/demo_palette.png -lavfi "fps=15,scale=960:-1:flags=lanczos [x]; [x][1:v] paletteuse" docs/demo.gif'
Write-Host ""
Write-Host '--- docs/dashboard.png ---'
Write-Host "  Dashboard URL: $DashboardUrl"
Write-Host "  Screenshot after at least one proactive decision (Win+Shift+S)."
Write-Host "  Save as docs/dashboard.png"
Write-Host ""

if (-not $NoBrowser) {
    Write-Host 'Opening dashboard in browser (start python -m proactivity.dashboard if it is down)...'
    Start-Process $DashboardUrl
}

Write-Host "Done. See docs/DEMO_ASSETS.md for macOS/Linux equivalents."
