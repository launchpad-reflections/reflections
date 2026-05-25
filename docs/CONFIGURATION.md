# Configuration

All runtime configuration is driven by environment variables. Copy `.env.example` to `.env` at the repo root. Python modules load it via `packages/reflections/env.py` (`load_env()`); the Bun camera server uses `dotenv`.

## MentraOS app server (required)

| Variable | Default | Description |
|----------|---------|-------------|
| `PACKAGE_NAME` | — | App package name from [console.mentra.glass](https://console.mentra.glass/) |
| `MENTRAOS_API_KEY` | — | API key from the MentraOS console |
| `PORT` | `3000` | Local HTTP port for the camera server; ngrok tunnels here |
| `WHIP_URL` | — | Glasses WHIP publish URL, e.g. `http://192.168.1.42:8889/live/glasses/whip`. **Must use your LAN IP**, not `127.0.0.1` |

## Streaming / viewer

| Variable | Default | Description |
|----------|---------|-------------|
| `WHEP_URL` | `http://127.0.0.1:8889/live/glasses/whep` | WHEP pull URL for `python -m apps.viewer` (localhost is correct) |
| `SPEAK_URL` | `http://127.0.0.1:{PORT}/speak` | TTS endpoint the proactivity agent POSTs to |
| `TRANSCRIPT_URL` | `http://127.0.0.1:{PORT}/transcripts` | SSE transcript stream (used when `USE_MENTRA_TRANSCRIPTION=true`) |
| `USE_MENTRA_TRANSCRIPTION` | `false` | If `true`, viewer reads cloud transcripts from Bun instead of local Soniox |
| `SHOW_INTERIM` | `false` | Print partial Soniox tokens to stdout |
| `TRANSCRIPT_LOG_PATH` | `transcript_updates.log` | Append-only log of transcript update events |

## Transcription & identity

| Variable | Default | Description |
|----------|---------|-------------|
| `SONIOX_API_KEY` | `""` | Soniox real-time STT key. If unset, Soniox processor is skipped |
| `USER_NAME` | `User` | Label for the wearer's speech in transcripts and captions |

## Proactivity & LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude API key — proactivity agent, memory snapshots, end-of-session name resolution |
| `PROACTIVITY_ENABLED` | `1` | Master switch for the proactivity classifier and agent. Set to `0`/`false`/`no` to skip the Qwen+LoRA download and disable Claude calls (transcription and ASD still run) |
| `GLASSES_GATE_THRESHOLD` | `0.25` | **Live gate.** Classifier P below this never reaches Claude. See [PROACTIVITY.md](PROACTIVITY.md) |
| `DEFAULT_PHRASE` | `Reflections speaker check` | Text spoken when pressing `p` in the viewer |
| `DASHBOARD_PORT` | `8766` | Port for `python -m proactivity.dashboard` |

> `REASONING_TRIGGER` (0.45) is **not** an environment variable — it is hard-coded in `packages/proactivity/classifier.py` and only affects test scripts, not the live glasses path.

## Default location (maps tools)

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LOCATION_NAME` | `Example City, CA` | Human-readable location bias for places search |
| `DEFAULT_LOCATION_LAT` | `0.0` | Latitude — **override before relying on maps tools** |
| `DEFAULT_LOCATION_LON` | `0.0` | Longitude — **override before relying on maps tools** |
| `DEFAULT_LOCATION_RADIUS_M` | `5000` | Search radius in meters |

## Google Maps (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_MAPS_API_KEY` | — | Enables `places_search`, `place_details`, `directions` tools. Enable Places API (New) + Directions API |

## Google Calendar (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_OAUTH_CLIENT_ID` | — | OAuth desktop client ID |
| `GOOGLE_OAUTH_CLIENT_SECRET` | — | OAuth client secret |
| `GOOGLE_OAUTH_REFRESH_TOKEN` | — | Long-lived refresh token from `python -m proactivity.calendar_auth` |

## Model / cache (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache root for Qwen 1.7B base download |
| `REFLECTIONS_LORA_MODEL_ID` | `rushilsaraf/qwen3-actionable-v2-adapter` | Hugging Face model ID for the proactivity LoRA adapter (classifier runtime + prefetch scripts) |

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Root log level via `packages/reflections/logging_config.py` |

## Files on disk (not env vars)

| Path | Purpose |
|------|---------|
| `memory.md` | Durable memory read on every classify; written on `s` press |
| `memory.example.md` | Starter template — copy to `memory.md` |
| `proactivity_decisions.log` | Agent gate/speak decisions |
| `proactivity_prompts.jsonl` | Full prompt log for dashboard |
| `models/face_gallery.npz` / `.json` | Persistent face identity gallery |

See [PRIVACY.md](PRIVACY.md) for what is stored locally vs sent to third-party APIs, and how to reset.

## Related docs

- [SETUP.md](SETUP.md) — minimal `.env` for first run
- [PROACTIVITY.md](PROACTIVITY.md) — threshold tuning
- [MODELS.md](MODELS.md) — weight files and download scripts
