# manager/

Per-user manager classes. Each manager handles one responsibility and is
instantiated inside the `User` class (`session/User.ts`).

| Class             | Responsibility                                                |
| ----------------- | ------------------------------------------------------------- |
| `SessionManager`  | Thin lookup — `Map<userId, User>` with get/create/remove      |
| `PhotoManager`    | Photo capture from button presses + in-memory cache           |
| `AudioManager`    | Optional TTS wrapper around `AppSession.audio` (scaffolding)  |
| `StorageManager`  | MentraOS Simple Storage helpers (scaffolding, currently unused) |
| `InputManager`    | Button presses and touchpad gestures (triggers `PhotoManager`)|

`/speak` does **not** go through `AudioManager` today — `CameraApp.ts`
calls `session.audio.speak()` directly on the SDK session so that one
HTTP call fans out to every active glasses session. `AudioManager` is
kept as a per-user wrapper that future code can use when speak-targeting
a specific user becomes useful.

Every manager (except `SessionManager`) receives a back-reference to its
`User` so it can access `this.user.appSession` and `this.user.userId`.

> Live transcription is no longer mediated by a `TranscriptionManager`.
> It is wired directly in `CameraApp.onSession()` (`../CameraApp.ts`),
> which subscribes to `session.events.onTranscription` and broadcasts to
> the SSE clients registered on `GET /transcripts`.
