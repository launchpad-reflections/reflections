# api/

HTTP handlers for the Express camera server. The server exposes the
minimal route surface needed to drive the Python viewer and proactivity
agent:

| Route               | Method | File / location                          |
| ------------------- | ------ | ---------------------------------------- |
| `/health`           | GET    | `health.ts` — liveness probe             |
| `/transcripts`      | GET    | inline in `../CameraApp.ts` — SSE stream |
| `/speak`            | POST   | inline in `../CameraApp.ts` — TTS proxy  |

Handlers under this directory are plain Express request handlers — wired
up directly in `apps/camera-server/src/server/CameraApp.ts`.

Unit tests live next to each handler (e.g. `health.test.ts`); end-to-end
HTTP coverage for the inline routes is in `../CameraApp.test.ts`.
