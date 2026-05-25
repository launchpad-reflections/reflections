# session/

Contains the `User` class — the per-user state container.

Each `User` composes all managers and holds the glasses `AppSession`. It's created by `SessionManager.getOrCreate(userId)` when a user connects and destroyed by `SessionManager.remove(userId)` on disconnect.

**Lifecycle:**
1. `new User(userId)` — instantiates all managers
2. `user.setAppSession(session)` — wires input and touch listeners; transcription is wired separately in `CameraApp.onSession()`
3. `user.clearAppSession()` — disconnects glasses, keeps cached photos
4. `user.cleanup()` — nukes everything
