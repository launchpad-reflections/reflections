"""Live dashboard for the proactivity prompt log.

Run:
    python -m proactivity.dashboard

Starts a local static server on http://127.0.0.1:8766/ that serves
proactivity/dashboard.html and tails proactivity_prompts.jsonl. Opens
the browser automatically.

Routes:
    GET  /           → proactivity/dashboard.html
    GET  /<file>     → static file from repo root (the dashboard fetches
                       /proactivity_prompts.jsonl as a sibling).
    POST /reset      → truncate proactivity_prompts.jsonl

Ctrl-C to quit.
"""

from __future__ import annotations

import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

from reflections.config import DASHBOARD_PORT, REPO_ROOT

from proactivity.promptlog import LOG_PATH, reset_log

DASHBOARD_HTML = Path(__file__).resolve().parent / "dashboard.html"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", ""):
            try:
                body = DASHBOARD_HTML.read_bytes()
            except FileNotFoundError:
                self.send_error(500, f"missing {DASHBOARD_HTML}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/reset":
            reset_log()
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:  # quieter access log
        sys.stderr.write(f"[dashboard] {fmt % args}\n")


def main() -> int:
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("127.0.0.1", DASHBOARD_PORT), Handler)
    server.daemon_threads = True

    url = f"http://127.0.0.1:{DASHBOARD_PORT}/"
    print(f"[dashboard] tailing {LOG_PATH}")
    print(f"[dashboard] {url}")
    print("[dashboard] ctrl+c to quit")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] bye")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
