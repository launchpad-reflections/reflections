# Agent guide — Reflections

Reference for AI coding agents working in this repository.

## Architecture

Glasses → **MediaMTX** (WHIP/WHEP on `:8889`, media on `:8189`) → **`python -m apps.viewer`** (aiortc viewer with processor pipeline)

- App server (Bun/TS) connects to MentraOS cloud and initiates streaming
- `packages/stream/source.py` runs aiortc on a background asyncio thread; `frames()` yields `VideoItem`/`AudioItem`
- **Transcription**: `SonioxProcessor` opens WebSocket to Soniox API; audio/finals/interim to stdout. Finals attributed to speaker via ASD. Low-latency `on_transcript_update(transcript)` event fires on speaker flip (200 ms debounced), sentence terminator (`.?!` or Soniox `<end>` endpoint token), or 0.5 s dwell with no growth; payload is the full cumulative `[(speaker, text), …]` including the in-progress sentence as the last entry. Emissions are signature-deduped with a 250 ms min-gap. Hook downstream processing (LLM reasoning, tool calls, UI agents) in `_on_transcript_update` in `apps/viewer/cli.py`.
- **Active speaker detection**: `ASDProcessor` uses LR-ASD model. YuNet detects faces; IoU tracker maintains 10-frame 112×112 crop buffers with 15→25 fps pulldown for 4:1 audio-to-video ratio. Batch inference on background thread; scores EMA-smoothed, drawn green/red/white. `who_spoke(t_start, t_end)` for retroactive attribution.
- **Identity**: `IdentityResolver` embeds faces (MobileFaceNet/ArcFace) into a persistent gallery (`models/face_gallery.npz` + `face_gallery.json`). New faces minted as "Person N"; matched faces reuse existing name. On exit, transcript is sent to Claude (Haiku) to resolve "Person N" labels to real names; gallery renamed before flush. Wearer attributed as `USER_NAME` from `.env`.

## Key files

| Path | Role |
|------|------|
| `apps/camera-server/src/server/CameraApp.ts` | Starts/stops streaming, handles reconnect, `/transcripts` SSE |
| `apps/viewer/cli.py` | WHEP viewer + processor pipeline (`python -m apps.viewer`) |
| `packages/stream/source.py`, `packages/stream/pipeline.py` | aiortc source and processor infrastructure |
| `packages/stream/processors/asd/`, `packages/stream/processors/soniox/` | Speaker detection, transcription |
| `packages/stream/processors/_identity.py`, `packages/stream/processors/asd/track.py` | Face embedding gallery, IoU tracker |
| `packages/reflections/config.py` | Central env-driven config (WHEP_URL, thresholds, `LORA_MODEL_ID`, default location) |
| `.env` | See `docs/CONFIGURATION.md` |
| `packages/third_party/lr_asd/`, `models/*.onnx` | ASD + face recognition weights |
| `models/face_gallery.npz`, `models/face_gallery.json` | Persistent face identity gallery (delete both to reset) |

## Running

```bash
# Terminal 1: MediaMTX
./mediamtx/mediamtx ./mediamtx/mediamtx.yml

# Terminal 2: App server (Bun)
bun run dev

# Terminal 3: Viewer
pip install -e . && python -m apps.viewer
```

See `docs/SETUP.md` for the full five-terminal workflow (ngrok + optional dashboard).

## Proactivity pipeline

- `packages/proactivity/agent/` runs on a daemon worker thread fed by `consider()` from `apps/viewer/cli.py`. Size-1 inbox evicts stale snapshots; freshness wins.
- **Two distinct thresholds — do not conflate them.**
  - `packages/reflections/config.py:GLASSES_GATE_THRESHOLD` (default **0.25**) is THE live gate. Sentences with classifier P below this never reach Claude. The agent reads it as `self.threshold`. This is the only knob the live glasses path consults.
  - `packages/proactivity/classifier.py:REASONING_TRIGGER` (0.45) is unrelated to the live path. It only decides whether the FULL classify path (used by `scripts/smoke_full_transcript.py`, `scripts/smoke_server.py`) generates a slow reasoning string. The live path uses `_label_only_classify` and ignores `label` for control flow.
- Other gates in `agent/worker.py` (override via `ProactivityAgent` kwargs): `min_consider_interval_s` (1.0), `min_claude_interval_s` (0.0), `min_speak_interval_s` (2.0), `repeat_text_window_s` (30.0), `recent_turns` (10).
- `_DEFAULT_TOOLS` in `agent/prompts.py` is the canonical tool-name list shown to the classifier (training vocabulary). Anthropic-side tools are built separately in `proactivity.tools.build_anthropic_tools()` (re-exported from `packages/proactivity/tools/__init__.py`: web_search + maps + calendar). Don't confuse the two lists.
- Proactivity classifier LoRA loads from Hugging Face Hub via `REFLECTIONS_LORA_MODEL_ID` (default `rushilsaraf/qwen3-actionable-v2-adapter` in `packages/reflections/config.py`). Prefetch with `./scripts/download_lora.sh`.
- `memory.md` is read on every classify, written only when user presses `s` in the viewer (`packages/proactivity/memory_agent.py`).
- Live debugging: `python -m proactivity.dashboard` opens a localhost dashboard that tails `proactivity_prompts.jsonl`.

## Notes

- `WHIP_URL` IP = laptop's WiFi IP (`ipconfig`); viewer uses `127.0.0.1`
- `rtmpUrl` field accepts any scheme; firmware routes on URL scheme
- ASD config: `detect_every_n`, `inference_hz`, `score_threshold`, `ema_alpha`, `display_timeout_sec`
- MediaMTX binary lives at `./mediamtx/mediamtx` inside the repo (gitignored, platform-specific). Run as `./mediamtx/mediamtx ./mediamtx/mediamtx.yml`.

## Documentation index

- `docs/ARCHITECTURE.md` — threading, data flow, mermaid diagrams
- `docs/SETUP.md` — local dev workflow
- `docs/CONFIGURATION.md` — all env vars
- `docs/PROACTIVITY.md` — thresholds, tools, dashboard
- `docs/MODELS.md` — weight downloads
- `docs/PRIVACY.md` — data handling
