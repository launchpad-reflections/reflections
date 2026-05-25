"""Local HTTP server for the actionability-classifier test UI.

Loads the Qwen 3 1.7B + LoRA classifier once at startup, then serves
test_ui.html and a /classify JSON endpoint so a browser can fire
arbitrary scenarios without re-paying the load cost.

Run:
    python scripts/smoke_server.py
Then open http://127.0.0.1:8765/ in any browser.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from proactivity.classifier import _default_location, classify, load_model  # noqa: E402
from reflections.config import REPO_ROOT
from reflections.env import load_env
from reflections.logging_config import setup_logging

UI_PATH = REPO_ROOT / "packages" / "proactivity" / "test_ui.html"
HOST = "127.0.0.1"
PORT = 8765
DEFAULT_LOCATION = _default_location()


def _parse_transcript(s: str) -> list[dict]:
    """Convert a multi-line transcript to a list of turn dicts. Accepts
    either 'Speaker: text' or 'Speaker | text' per line. Strips
    surrounding [brackets] from the speaker label so the existing
    transcript_updates.log format pastes in cleanly."""
    turns: list[dict] = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            speaker, _, text = line.partition("|")
        elif ":" in line:
            speaker, _, text = line.partition(":")
        else:
            continue
        speaker = speaker.strip().lstrip("[").rstrip("]")
        text = text.strip()
        if speaker and text:
            turns.append({"speaker": speaker, "text": text})
    if turns:
        turns[-1]["is_target"] = True
    return turns


def _build_example(payload: dict) -> dict:
    turns = _parse_transcript(payload.get("transcript", ""))
    if not turns:
        raise ValueError("transcript is empty (need at least one Speaker: text line)")

    memory = (payload.get("memory") or "").strip()
    memory_summaries = (
        [{"timestamp_approx": "current_session", "summary": memory}] if memory else []
    )

    tools = payload.get("tools") or ["send_message"]
    entity_list = payload.get("entity_list") or []

    return {
        "id": "ui_test",
        "transcript": {
            "turns": turns,
            "target_speaker": turns[-1]["speaker"],
            "target_index": len(turns) - 1,
        },
        "memory_summaries": memory_summaries,
        "available_tools": tools,
        "location": payload.get("location") or DEFAULT_LOCATION,
        "entity_list": entity_list,
        "label": 0,
        "reasoning": "(test)",
        "metadata": {
            "category": "memory_dependent",
            "subcategory": "ui_test",
            "difficulty": "medium",
            "signals_used": [],
            "action_type": None,
        },
    }


class Handler(BaseHTTPRequestHandler):
    # Set on the class by main() once the model is loaded.
    model = None
    tokenizer = None
    device = None
    t0_id = None
    t1_id = None

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib requires this name)
        if self.path == "/":
            try:
                body = UI_PATH.read_bytes()
            except FileNotFoundError:
                self._send_json(500, {"error": f"missing {UI_PATH}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/classify":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            example = _build_example(payload)

            t0 = time.monotonic()
            p, label, reasoning = classify(
                self.model,
                self.tokenizer,
                self.device,
                self.t0_id,
                self.t1_id,
                example,
            )
            inference_ms = (time.monotonic() - t0) * 1000.0

            self._send_json(
                200,
                {
                    "p": float(p),
                    "label": int(label),
                    "reasoning": (reasoning.strip() if reasoning else ""),
                    "inference_ms": inference_ms,
                    "turns": len(example["transcript"]["turns"]),
                },
            )
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args) -> None:  # quieter access log
        sys.stderr.write(f"[server] {fmt % args}\n")


def main() -> None:
    load_env()
    setup_logging()
    print("[boot] loading model (downloads Qwen 3 1.7B on first run)...")
    Handler.model, Handler.tokenizer, Handler.device, Handler.t0_id, Handler.t1_id = load_model()

    server = HTTPServer((HOST, PORT), Handler)
    print(f"\n[boot] ready — open http://{HOST}:{PORT}/")
    print("[boot] ctrl+c to quit")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[boot] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
