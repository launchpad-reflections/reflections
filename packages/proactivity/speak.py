"""Tell the glasses to speak a phrase via the Bun app server.

Sends a POST to /speak on the local Bun server, which invokes
session.audio.speak() on every active glasses session. Triggered by
pressing `p` in the apps.viewer window; also runnable standalone:

    python -m proactivity.speak                 # default phrase
    python -m proactivity.speak "Hello there"   # custom phrase
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request

from reflections.config import DEFAULT_PHRASE, SPEAK_URL

logger = logging.getLogger(__name__)


def speak(text: str = DEFAULT_PHRASE, url: str = SPEAK_URL, timeout: float = 10.0) -> bool:
    """Ask the glasses to speak `text`. Returns True on HTTP 2xx.

    Runs synchronously — the HTTP call waits for the TTS request to
    be accepted by MentraOS cloud (not for playback to finish). Call
    from a background thread if you don't want to block."""
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if 200 <= resp.status < 300:
                logger.info("[speak] glasses said: %r", text)
                return True
            logger.error("[speak] HTTP %s: %s", resp.status, body)
            return False
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        logger.error("[speak] HTTP %s: %s", e.code, body or e.reason)
        return False
    except Exception as e:
        logger.error("[speak] request failed: %s", e)
        return False


if __name__ == "__main__":
    phrase = " ".join(sys.argv[1:]) or DEFAULT_PHRASE
    ok = speak(phrase)
    sys.exit(0 if ok else 1)
