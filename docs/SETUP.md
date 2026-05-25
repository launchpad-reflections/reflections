# Setup

End-to-end guide for running Reflections on your laptop with MentraOS glasses on the same WiFi network.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| [Bun](https://bun.sh/) | ≥ 1.1 | Runs `apps/camera-server` |
| Python | ≥ 3.10 | Required for proactivity classifier (`peft==0.19.x`) |
| [ngrok](https://ngrok.com/download) | any | Free account; `ngrok config add-authtoken <TOKEN>` once |
| MediaMTX | latest release | Binary in `./mediamtx/mediamtx` (or `mediamtx.exe` on Windows) |

Verify:

```bash
bun --version
python --version
ngrok version
```

**Windows (PowerShell):**

```powershell
.\mediamtx\mediamtx.exe --version
```

**macOS / Linux:**

```bash
./mediamtx/mediamtx --version
```

Download MediaMTX from [github.com/bluenviron/mediamtx/releases](https://github.com/bluenviron/mediamtx/releases) and place the binary in `mediamtx/` (config `mediamtx/mediamtx.yml` is checked in).

## 1. Clone and install

```bash
git clone <this-repo-url> reflections
cd reflections
bun install
```

**Python (venv recommended):**

```bash
python -m venv venv
pip install -e .
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

Editable install (`pip install -e .`) exposes `apps`, `packages/stream`, `packages/proactivity`, and `packages/reflections` as importable modules.

Contributors should install dev dependencies:

```bash
pip install -e ".[dev]"
```

### Verify installation

After clone and install, run the automated sanity check (no glasses, MediaMTX, or ngrok required):

**Windows (PowerShell):**

```powershell
.\scripts\verify_fresh_setup.ps1
```

**macOS / Linux:**

```bash
chmod +x scripts/verify_fresh_setup.sh
./scripts/verify_fresh_setup.sh
```

The script checks Python ≥ 3.10, installs `pip install -e ".[dev]"`, runs `pytest`, Bun typecheck/tests, `python -m apps.viewer --help-only`, and core Python imports. It prints a pass/fail summary and exits non-zero on failure.

## 2. Model weights

Download ONNX models and optionally prefetch the LoRA adapter. See [MODELS.md](MODELS.md) for full instructions.

Minimum manual step: place these in `models/`:

- `face_detection_yunet_2023mar.onnx` (YuNet face detection)
- `w600k_mbf.onnx` (MobileFaceNet / ArcFace embeddings)

LR-ASD weights ship under `packages/third_party/lr_asd/weights/`. The Qwen 1.7B base auto-downloads on first classifier load (~3.4 GB).

## 3. MentraOS app registration

1. Sign in at [console.mentra.glass](https://console.mentra.glass/).
2. Create an app; note the **Package Name** and **API Key**.
3. Enable **camera** and **microphone** permissions.
4. Leave **Public URL** empty until ngrok is running (step 5).

## 4. Environment variables

```bash
cp .env.example .env
```

Edit `.env` — see [CONFIGURATION.md](CONFIGURATION.md) for every variable. Minimum required:

```
PACKAGE_NAME=com.yourname.reflections
MENTRAOS_API_KEY=...
WHIP_URL=http://<YOUR_LAN_IP>:8889/live/glasses/whip
SONIOX_API_KEY=...
ANTHROPIC_API_KEY=...
USER_NAME=Your Name
```

### Find your LAN IP

Glasses connect to MediaMTX **directly over WiFi** — use your laptop's LAN address, not `127.0.0.1` or the ngrok host.

| OS | Command |
|----|---------|
| Windows | `ipconfig` → IPv4 under active WiFi adapter |
| macOS | `ipconfig getifaddr en0` |
| Linux | `ip addr show` |

Re-check after changing networks; update `WHIP_URL` before restarting the camera server.

## 5. Five-terminal workflow

Run these in separate terminals from the repo root. Keep all five open during a session.

### Terminal 1 — ngrok

```bash
ngrok http 3000
```

Copy the `https://….ngrok-free.app` URL into your app's **Public URL** in the MentraOS console. Restarting ngrok on the free plan generates a new URL — update the console each time.

### Terminal 2 — MediaMTX

**Windows (PowerShell):**

```powershell
.\mediamtx\mediamtx.exe .\mediamtx\mediamtx.yml
```

**macOS / Linux:**

```bash
./mediamtx/mediamtx ./mediamtx/mediamtx.yml
```

Expect WebRTC on `:8889` and UDP media on `:8189`. Allow through the firewall on **private networks**.

### Terminal 3 — camera server

**Windows (PowerShell):**

```powershell
$env:NODE_ENV = "development"
bun --watch apps/camera-server/src/index.ts
```

**macOS / Linux:**

```bash
NODE_ENV=development bun --watch apps/camera-server/src/index.ts
```

Or use the package script: `bun run dev` (same entrypoint: `apps/camera-server/src/index.ts`).

You should see `Listening on :3000`. After launching the app on glasses: `Camera session started` → `Starting WHIP stream`.

### Terminal 4 — viewer

Activate your venv if needed, then:

```bash
python -m apps.viewer
```

Connects to `WHEP_URL` (default `http://127.0.0.1:8889/live/glasses/whep`). Opens an OpenCV window titled **"Mentra Live"** with bounding boxes and captions.

### Terminal 5 — proactivity dashboard (optional, recommended for tuning)

```bash
python -m proactivity.dashboard
```

Opens `http://127.0.0.1:8766/` and tails `proactivity_prompts.jsonl` — every classifier prompt, Claude request/response, and tool call. See [PROACTIVITY.md](PROACTIVITY.md).

## 6. Launch on glasses

1. Open the MentraOS phone app; ensure glasses are paired and on the same WiFi.
2. Launch your app from the app list.
3. Within ~1 s: ngrok request → Bun server session log → MediaMTX publish → viewer window opens.

## Keyboard controls

Click the **Mentra Live** window to focus it first.

| Key | Action |
|-----|--------|
| `s` | Snapshot transcript delta to `memory.md` via Claude |
| `p` | Speak the default phrase through glasses TTS |
| `m` | Mute / unmute proactivity TTS (classifier still runs) |
| `q` | Quit cleanly — name resolution + face gallery flush |

## Troubleshooting

**No video in viewer**

- Wrong `WHIP_URL` LAN IP, or glasses on a different network.
- Firewall blocking UDP `:8189`.

**`PACKAGE_NAME and MENTRAOS_API_KEY must be set`**

- Missing or empty `.env`.

**MentraOS can't reach the app**

- ngrok down or stale Public URL in console.

**No transcripts**

- `SONIOX_API_KEY` missing — viewer skips Soniox processor.
- Or set `USE_MENTRA_TRANSCRIPTION=true` to use Bun `/transcripts` SSE instead.

**Stream drops**

- WiFi flake or LAN IP changed. Restart camera server after fixing `WHIP_URL`.

**Port conflicts**

- Change `PORT` in `.env` and the ngrok command together.
- For MediaMTX ports, edit `mediamtx/mediamtx.yml` and mirror changes in `WHIP_URL` / `WHEP_URL`.

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — threading and data flow
- [CONFIGURATION.md](CONFIGURATION.md) — all environment variables
- [MODELS.md](MODELS.md) — weight downloads
- [PRIVACY.md](PRIVACY.md) — data handling and reset
