# Demo assets

This repo ships two optional marketing assets under `docs/` for the README and release notes:

| File | Purpose | Target |
|------|---------|--------|
| `docs/demo.gif` | Live viewer with ASD bounding boxes, speaker captions, and a proactive TTS reply | ~10 s, ≤ 5 MB |
| `docs/dashboard.png` | Proactivity dashboard showing classifier + Claude decisions | Single screenshot |

These files are **not** generated in CI — they require a live glasses session on a developer machine. The first public release ships without them; capture and add them in a follow-up commit whenever a glasses session is convenient.

## Privacy and content rules

> **Do not use anything from the `recordings/` folder.**

MediaMTX can write session recordings to `recordings/` (see `mediamtx/mediamtx.yml`). That directory may contain **real microphone audio and identifiable speech**. It is gitignored for a reason.

For demo capture:

- Use **synthetic or clearly fictional speech only** — read a short script yourself, use a TTS voice on another device, or have a colleague speak lines you write for the demo.
- Do not commit `.env`, API keys, face galleries, `memory.md`, or runtime logs.
- Blur or crop anything sensitive (names on screen, URLs in the dashboard, private calendar data from tool calls).
- Prefer a neutral background and generic place names in spoken lines (e.g. “Find ramen nearby” rather than a real address).

## Prerequisites

Complete the [five-terminal setup](SETUP.md):

1. `ngrok http 3000`
2. `./mediamtx/mediamtx ./mediamtx/mediamtx.yml` (or `mediamtx.exe` on Windows)
3. `bun run dev`
4. `python -m apps.viewer` — opens the **Mentra Live** OpenCV window
5. `python -m proactivity.dashboard` — opens `http://127.0.0.1:8766/`

Launch the app on your glasses from the MentraOS phone app. Confirm bounding boxes and captions appear in **Mentra Live**.

Optional helpers:

```bash
# Windows
.\scripts\capture_demo.ps1

# macOS / Linux
./scripts/capture_demo.sh
```

These scripts print capture commands and open the dashboard URL; they do **not** require a live stream by themselves.

## Suggested demo script (~10 s)

Use fictional dialogue so the proactive pipeline fires. Example:

1. **Speaker (you or a colleague, on camera):** “Hey Reflections, what’s a good ramen place around here?”
2. Wait for green/red ASD boxes and live captions to update.
3. Let the classifier gate pass the turn and Claude speak through the glasses (proactive TTS).
4. Keep the **Mentra Live** window in frame for the full clip.

Tips:

- Click the **Mentra Live** window before using keyboard shortcuts.
- Press **`m`** to unmute proactivity if TTS is muted.
- Lower `GLASSES_GATE_THRESHOLD` temporarily (e.g. `0.20`) if the gate is too strict for a staged line — see [PROACTIVITY.md](PROACTIVITY.md).
- Press **`p`** only for a manual TTS smoke test; the GIF should show a **proactive** reply, not just the default phrase.

## Capture `docs/demo.gif`

Record the **Mentra Live** window (or a cropped region around it), not the full desktop with secrets visible.

### Windows

**Record (pick one):**

- **Xbox Game Bar:** `Win+G` → Capture → Record selected area over the viewer window.
- **ffmpeg (gdigrab):** adjust `-offset_x`/`-offset_y`/`-video_size` to your window position.

```powershell
# Example: record 10 s from the desktop region where Mentra Live appears
ffmpeg -f gdigrab -framerate 30 -offset_x 100 -offset_y 100 -video_size 1280x720 -t 10 -i desktop docs/demo_raw.mp4
```

**Convert to GIF:**

```powershell
# Palette-based GIF (good quality, smaller than naive conversion)
ffmpeg -i docs/demo_raw.mp4 -vf "fps=15,scale=960:-1:flags=lanczos,palettegen" docs/demo_palette.png
ffmpeg -i docs/demo_raw.mp4 -i docs/demo_palette.png -lavfi "fps=15,scale=960:-1:flags=lanczos [x]; [x][1:v] paletteuse" docs/demo.gif
Remove-Item docs/demo_raw.mp4, docs/demo_palette.png
```

Alternatives: [ScreenToGif](https://www.screentogif.com/) (export directly to `docs/demo.gif`).

### macOS

**Record:**

- **QuickTime:** File → New Screen Recording → drag a region over **Mentra Live**.
- **ffmpeg (avfoundation):** list devices with `ffmpeg -f avfoundation -list_devices true -i ""`, then capture screen index `1` (often the main display).

```bash
ffmpeg -f avfoundation -framerate 30 -pixel_format uyvy422 -i "1:none" -t 10 docs/demo_raw.mp4
```

**Convert to GIF:**

```bash
ffmpeg -i docs/demo_raw.mp4 -vf "fps=15,scale=960:-1:flags=lanczos,palettegen" docs/demo_palette.png
ffmpeg -i docs/demo_raw.mp4 -i docs/demo_palette.png -lavfi "fps=15,scale=960:-1:flags=lanczos [x]; [x][1:v] paletteuse" docs/demo.gif
rm docs/demo_raw.mp4 docs/demo_palette.png
```

Trim to ~10 s and aim for ≤ 5 MB. Reduce `fps` or `scale` if the file is too large.

## Capture `docs/dashboard.png`

1. With the viewer running and at least one proactive decision logged, open `http://127.0.0.1:8766/` (or run `python -m proactivity.dashboard`).
2. Confirm the log shows classifier prompts, gate scores, and a Claude speak/skip decision.
3. Crop out browser chrome if needed; redact API snippets or personal data.
4. Save as **`docs/dashboard.png`**.

**Windows:** `Win+Shift+S` → window snip, or Snipping Tool.

**macOS:** `Cmd+Shift+4` then `Space` to capture the browser window, or `Cmd+Shift+4` for a region.

```bash
# macOS — full window by title (if visible)
screencapture -l$(osascript -e 'tell app "Safari" to id of window 1') docs/dashboard.png
```

## Check in before release

1. Place files at `docs/demo.gif` and `docs/dashboard.png`.
2. Update [README.md](../README.md): replace the “demo pending” note with the GIF embed and dashboard image if desired.
3. Verify assets in a clean clone — paths must be relative (`docs/demo.gif`).
4. Never add `recordings/` or other gitignored runtime data to the commit.

## Related docs

- [SETUP.md](SETUP.md) — full local workflow
- [PROACTIVITY.md](PROACTIVITY.md) — gate thresholds and dashboard
- [PRIVACY.md](PRIVACY.md) — data handling
