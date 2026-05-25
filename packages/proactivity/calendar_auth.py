"""One-time Google Calendar OAuth bootstrap.

Walks through the installed-app OAuth flow with no third-party deps:
spins up a localhost callback server, opens the consent page in a
browser, captures the authorization code, exchanges it for a refresh
token, and prints the refresh token for pasting into `.env`.

Usage:
    # set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in .env
    # (or in the shell), then run:
    python -m proactivity.calendar_auth

The script never touches `.env` itself — copy the printed refresh
token into the file by hand.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import secrets
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from reflections.env import load_env
from reflections.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Read+write events on the wearer's calendars. Note: Google's scope
# model has no "no-delete" tier — restricting deletes is enforced by
# our tool surface, not by the OAuth scope.
SCOPE = "https://www.googleapis.com/auth/calendar.events"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CodeCatcher(http.server.BaseHTTPRequestHandler):
    """Captures the ?code=... query param from Google's redirect."""

    captured: dict[str, str | None] = {"code": None, "state": None, "error": None}

    def do_GET(self):  # noqa: N802
        parts = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parts.query)
        _CodeCatcher.captured["code"] = (qs.get("code") or [None])[0]
        _CodeCatcher.captured["state"] = (qs.get("state") or [None])[0]
        _CodeCatcher.captured["error"] = (qs.get("error") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        body = (
            "<html><body><h2>Authorization received.</h2>"
            "<p>You can close this tab and return to the terminal.</p>"
            "</body></html>"
        )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *_args, **_kwargs):  # silence default access log
        return


def _exchange_code(code: str, redirect_uri: str, client_id: str, client_secret: str) -> dict:
    body = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    load_env()
    setup_logging()
    cid = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    sec = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
    if not cid or not sec:
        logger.error(
            "GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET not set.\n"
            "Add them to .env (see .env.example for setup steps).",
        )
        return 2

    port = _free_port()
    redirect_uri = f"http://127.0.0.1:{port}"
    state = secrets.token_urlsafe(16)
    auth_params = urllib.parse.urlencode(
        {
            "client_id": cid,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPE,
            # access_type=offline + prompt=consent forces Google to issue
            # a refresh token even on subsequent runs against the same
            # account — without prompt=consent you only get one once.
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
    )
    auth_url = f"{AUTH_URL}?{auth_params}"

    print(f"[calendar_auth] listening on {redirect_uri}")
    print("[calendar_auth] opening browser for consent...")
    print(f"[calendar_auth] (if it doesn't open, paste this URL in:)\n  {auth_url}\n")
    webbrowser.open(auth_url)

    server = http.server.HTTPServer(("127.0.0.1", port), _CodeCatcher)
    try:
        server.handle_request()  # blocks until Google redirects to us once
    except KeyboardInterrupt:
        print("\n[calendar_auth] aborted")
        return 130
    finally:
        server.server_close()

    captured = _CodeCatcher.captured
    if captured["error"]:
        logger.error("[calendar_auth] consent failed: %s", captured["error"])
        return 1
    if captured["state"] != state:
        logger.error("[calendar_auth] state mismatch — possible CSRF")
        return 1
    code = captured["code"]
    if not code:
        logger.error("[calendar_auth] no authorization code received")
        return 1

    try:
        tokens = _exchange_code(code, redirect_uri, cid, sec)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error("[calendar_auth] token exchange HTTP %s: %s", e.code, body)
        return 1

    refresh = tokens.get("refresh_token")
    if not refresh:
        logger.error(
            "[calendar_auth] Google did not return a refresh_token. This "
            "usually means you've already authorized this client for the "
            "same account before. Revoke at "
            "https://myaccount.google.com/permissions and re-run.",
        )
        return 1

    print()
    print("=" * 60)
    print("Paste this into .env:")
    print()
    print(f"GOOGLE_OAUTH_REFRESH_TOKEN={refresh}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
