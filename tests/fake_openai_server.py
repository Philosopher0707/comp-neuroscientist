"""
Fake OpenAI-compatible streaming server for end-to-end protocol tests.

Speaks just enough of the `/v1/chat/completions` SSE protocol for the real
agent loop in `claude_agent_sdk._run_agent_loop` to run against it, with no
network access and no Ollama dependency.

A "script" is a list of turns. Each turn is either:
  - {"text": "..."}                      -> stream content deltas
  - {"tool": (name, args_dict)}          -> stream a tool_call delta
Turns are served in order; each HTTP request consumes one turn. When the
script is exhausted the server keeps serving an empty final turn, so a
loop that over-runs is still well-defined rather than hanging.

The server also records every request body it received, so tests can assert
on what the agent actually sent (e.g. that tool results were fed back in).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _sse(payload: dict[str, Any]) -> bytes:
    """Encode one OpenAI-style SSE chunk."""
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    return _sse({
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake-model",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    })


class FakeOpenAIServer:
    """A scripted OpenAI-compatible SSE server, running on a background thread.

    Usage:
        with FakeOpenAIServer([{"text": "hi"}]) as srv:
            base_url = srv.base_url
        # ... point ANTHROPIC_BASE_URL at base_url ...
    """

    def __init__(self, script: list[dict[str, Any]], port: int = 0) -> None:
        self._script = list(script)
        self._requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._index = [0]
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port = port

    # ─── lifecycle ───────────────────────────────────────────────

    def start(self) -> "FakeOpenAIServer":
        script = self._script
        lock = self._lock
        requests_sink = self._requests
        index_sink = self._index

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # silence stderr spam
                pass

            def do_POST(self) -> None:  # noqa: N802 (http.server API)
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode() or "{}")
                except json.JSONDecodeError:
                    body = {}

                with lock:
                    requests_sink.append(body)
                    idx = index_sink[0]
                    index_sink[0] = idx + 1

                turn = script[idx] if idx < len(script) else {}

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                out = bytearray()

                if "text" in turn:
                    text = turn["text"]
                    # Split into a few deltas to exercise the streaming path
                    # and prove the agent accumulates them.
                    for piece in _chunks(text, 3):
                        out += _chunk({"content": piece})
                    out += _chunk({}, finish="stop")

                elif "tool" in turn:
                    name, args = turn["tool"]
                    # A tool call arrives as one delta with id+name+arguments,
                    # matching how Ollama streams tool calls.
                    out += _chunk({
                        "tool_calls": [{
                            "index": 0,
                            "id": f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }]
                    })
                    out += _chunk({}, finish="tool_calls")

                else:
                    # Exhausted script: terminate cleanly with no content.
                    out += _chunk({}, finish="stop")

                self.wfile.write(bytes(out))
                self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                # The OpenAI client may probe /v1/models; answer benignly.
                payload = json.dumps({"object": "list", "data": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self._port), Handler)
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "FakeOpenAIServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ─── inspection ──────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        """Base URL to hand to the OpenAI client (without /v1)."""
        return f"http://127.0.0.1:{self._port}"

    @property
    def requests(self) -> list[dict[str, Any]]:
        """Every request body the agent sent, in order."""
        with self._lock:
            return list(self._requests)

    @property
    def request_count(self) -> int:
        with self._lock:
            return len(self._requests)


def _chunks(text: str, n: int) -> list[str]:
    """Split text into ~n pieces, preserving all characters."""
    if not text:
        return []
    size = max(1, len(text) // n)
    pieces = [text[i:i + size] for i in range(0, len(text), size)]
    return pieces
