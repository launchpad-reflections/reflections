# Privacy

Reflections processes live audio and video from smart glasses on your local machine. This document describes what data is captured, where it is stored, and how to reset it.

## Data captured during a session

### Transcripts

Speech is transcribed in real time via the Soniox API. Transcript text (with speaker labels from active speaker detection) is used by the proactivity pipeline and may be logged locally when debugging is enabled.

### Face embeddings

The identity resolver embeds detected faces (MobileFaceNet/ArcFace) and stores them in a persistent gallery so recurring people can be labeled consistently across sessions. Gallery files:

- `models/face_gallery.npz`
- `models/face_gallery.json`

These files contain numeric embeddings and display names (e.g. "Person 1", resolved names after session exit). They are gitignored and never committed.

### memory.md

Pressing `s` in the viewer snapshots durable facts about people in your conversations into `memory.md` at the repo root. This file is read on every proactivity classify call and is intended to stay on your machine only.

Copy `memory.example.md` to `memory.md` to seed a blank gallery, or start with an empty file.

### Debug logs (optional)

When running locally, these files may be created in the repo root:

- `proactivity_prompts.jsonl` — classifier and Claude prompts/responses
- `proactivity_decisions.log` — gate and speak decisions
- `transcript_updates.log` — transcript update events

All are gitignored.

## Local storage summary

| Artifact | Location | Sent off-device? |
|----------|----------|------------------|
| Transcripts (live) | stdout / optional logs | Soniox API (audio stream) |
| Face gallery | `models/face_gallery.*` | No |
| Memory | `memory.md` | No (local file) |
| Proactivity debug | `proactivity_*.jsonl`, `*.log` | No |
| API calls | Anthropic, Google Maps/Calendar, web search | Yes, when tools fire |

Configure API keys in `.env`. Do not commit `.env`. See [CONFIGURATION.md](CONFIGURATION.md).

## Reset procedure

To wipe all locally accumulated identity and memory data:

```bash
# Remove face gallery
rm -f models/face_gallery.npz models/face_gallery.json

# Remove memory and debug artifacts
rm -f memory.md proactivity_prompts.jsonl proactivity_decisions.log transcript_updates.log
```

**Windows (PowerShell):**

```powershell
Remove-Item -Force models/face_gallery.npz, models/face_gallery.json -ErrorAction SilentlyContinue
Remove-Item -Force memory.md, proactivity_prompts.jsonl, proactivity_decisions.log, transcript_updates.log -ErrorAction SilentlyContinue
```

Restart the viewer (`python -m apps.viewer`) for a clean session. New faces will be minted as "Person N" again; copy `memory.example.md` to `memory.md` if you want a starter template.

## Default location

Maps and directions tools bias searches to a default location configured via environment variables in `.env`:

- `DEFAULT_LOCATION_NAME` (default: `Example City, CA`)
- `DEFAULT_LOCATION_LAT` / `DEFAULT_LOCATION_LON`

Set these to your area before using place search or directions in production. See [CONFIGURATION.md](CONFIGURATION.md).

## Related docs

- [MODELS.md](MODELS.md) — face gallery and model artifacts
- [PROACTIVITY.md](PROACTIVITY.md) — when data is sent to Anthropic and Google APIs
- [SECURITY.md](../SECURITY.md) — vulnerability reporting
