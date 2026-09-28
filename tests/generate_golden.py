"""Generate tests/testdata/protocol_golden.ndjson from a real agent run.

The fixture is what both sides agree on: the Go tests decode it, and the Python
E2E suite asserts a live run can still produce every event type it declares.
Regenerate with:  python tests/generate_golden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from fake_openai_server import FakeOpenAIServer  # noqa: E402
from comp_neuroscientist.agent import consume_stream, _setup_environment  # noqa: E402
from claude_agent_sdk import query  # noqa: E402
import os  # noqa: E402

GOLDEN = HERE / "testdata" / "protocol_golden.ndjson"


def capture(script):
    """Run the agent in-process and capture the exact JSON lines it emits."""
    lines: list[str] = []

    import comp_neuroscientist.agent as agent_mod
    real_emit = agent_mod._emit_json

    def capture_emit(payload):
        lines.append(json.dumps(payload, sort_keys=True))
        real_emit(payload)

    agent_mod._emit_json = capture_emit

    srv = FakeOpenAIServer(script).start()
    os.environ["ANTHROPIC_BASE_URL"] = srv.base_url
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        options = _setup_environment("analyze the data")
        stream = query("analyze the data", options)
        consume_stream(stream, json_mode=True)
    finally:
        srv.stop()
        agent_mod._emit_json = real_emit
    return lines


def main() -> None:
    probe = HERE / "testdata" / "golden_probe.txt"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.unlink(missing_ok=True)

    lines = capture([
        {"tool": ("Write", {"file_path": str(probe), "content": "x"})},
        {"text": "Analysis complete."},
    ])

    # The error event is not produced by a happy-path run; it is part of the
    # documented contract and the Go decoder handles it, so include a
    # representative line explicitly.
    lines.append(json.dumps(
        {"type": "error", "message": "API connection failed"}, sort_keys=True))

    # Absolute paths differ per machine and would make the fixture churn on
    # every checkout; rewrite them to a stable placeholder so the shared
    # fixture stays byte-identical everywhere.
    normalized = [
        line_text.replace(str(HERE), "<tests>").replace(str(HERE.parent), "<repo>")
        for line_text in lines
    ]

    GOLDEN.write_text("\n".join(normalized) + "\n")
    probe.unlink(missing_ok=True)
    print(f"wrote {len(normalized)} lines to {GOLDEN}")
    for line_text in normalized:
        print("  ", line_text)


if __name__ == "__main__":
    main()
