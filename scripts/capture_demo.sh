#!/usr/bin/env bash
# Demo asset capture helper (documentation only — no live stream required).
#
# Prints ffmpeg / GIF steps for docs/demo.gif and docs/dashboard.png.
# Optionally opens the proactivity dashboard in your browser.
#
# Usage:
#   ./scripts/capture_demo.sh              # print instructions + open dashboard
#   ./scripts/capture_demo.sh --no-browser # print instructions only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OPEN_BROWSER=1
if [[ "${1:-}" == "--no-browser" ]]; then
  OPEN_BROWSER=0
fi

DASHBOARD_PORT="${DASHBOARD_PORT:-$(
  python -c "from reflections.config import DASHBOARD_PORT; print(DASHBOARD_PORT)" 2>/dev/null || echo 8766
)}"
DASHBOARD_URL="http://127.0.0.1:${DASHBOARD_PORT}/"

echo ""
echo "=== Reflections demo asset capture (macOS / Linux) ==="
echo ""
echo "Full guide: docs/DEMO_ASSETS.md"
echo ""
echo "WARNING: Do NOT use files from recordings/ — they may contain real audio."
echo "         Use synthetic or fictional speech only."
echo ""
echo "--- Prerequisites (live session) ---"
echo "  1. ngrok http 3000"
echo "  2. ./mediamtx/mediamtx ./mediamtx/mediamtx.yml"
echo "  3. bun run dev"
echo "  4. python -m apps.viewer          # Mentra Live window"
echo "  5. python -m proactivity.dashboard"
echo ""
echo "--- docs/demo.gif (~10 s viewer + proactive TTS) ---"
echo "  Record the Mentra Live OpenCV window (QuickTime or ffmpeg avfoundation)."
echo ""
echo "  ffmpeg example (screen index may differ — run: ffmpeg -f avfoundation -list_devices true -i \"\"):"
echo '    ffmpeg -f avfoundation -framerate 30 -pixel_format uyvy422 -i "1:none" -t 10 docs/demo_raw.mp4'
echo ""
echo "  Convert to GIF:"
echo '    ffmpeg -i docs/demo_raw.mp4 -vf "fps=15,scale=960:-1:flags=lanczos,palettegen" docs/demo_palette.png'
echo '    ffmpeg -i docs/demo_raw.mp4 -i docs/demo_palette.png -lavfi "fps=15,scale=960:-1:flags=lanczos [x]; [x][1:v] paletteuse" docs/demo.gif'
echo ""
echo "--- docs/dashboard.png ---"
echo "  Dashboard URL: ${DASHBOARD_URL}"
echo "  Screenshot after at least one proactive decision (Cmd+Shift+4 on macOS)."
echo "  Save as docs/dashboard.png"
echo ""

if [[ "${OPEN_BROWSER}" -eq 1 ]]; then
  echo "Opening dashboard in browser (start python -m proactivity.dashboard if it is down)..."
  if command -v open >/dev/null 2>&1; then
    open "${DASHBOARD_URL}"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${DASHBOARD_URL}"
  else
    echo "(No open/xdg-open — visit ${DASHBOARD_URL} manually)"
  fi
fi

echo "Done. See docs/DEMO_ASSETS.md for Windows equivalents."
