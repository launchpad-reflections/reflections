# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| `main` branch | ✅ |

## Reporting a vulnerability

**Please do not open public GitHub issues for security vulnerabilities.**

Preferred channel: open a private
[GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (`Security` tab → `Report a vulnerability`).

If you cannot use Security Advisories, email the address listed under the
repository's `Contact` section on GitHub. Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to acknowledge reports within 48 hours and will coordinate disclosure once a fix is available.

## Scope notes

Reflections processes live audio and video locally and sends data to third-party APIs when configured:

- **Soniox** — audio for transcription
- **Anthropic** — transcript and memory for LLM calls
- **Google Maps / Calendar** — when proactivity tools are enabled

Review [docs/PRIVACY.md](docs/PRIVACY.md) for data handling details.

### Secrets

- Never commit `.env`, OAuth refresh tokens, or API keys.
- Rotate keys immediately if accidentally exposed.
- Face gallery files (`models/face_gallery.*`) contain biometric embeddings — treat as sensitive local data.

### Local services

The proactivity dashboard (`python -m proactivity.dashboard`) binds to `127.0.0.1` by default. Do not expose it to untrusted networks without authentication.

## Safe defaults

- `GLASSES_GATE_THRESHOLD` gates outbound Claude calls on the live path.
- Press `m` in the viewer to mute TTS without stopping classification.
- Delete `models/face_gallery.*` and `memory.md` to reset local identity and memory data (see [docs/PRIVACY.md](docs/PRIVACY.md)).
