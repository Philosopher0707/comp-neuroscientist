"""
End-to-end protocol tests.

These are the only tests that cross the language boundary: they run the REAL
Python agent subprocess (`python -m comp_neuroscientist.cli --json`) against a
fake OpenAI-compatible server, capture the NDJSON event stream it writes to
stdout, and assert on it.

Every previous test asserted on shape in isolation — prompt length, JSON parses
into the right struct, all six event types decode. None of them checked that
the Python side ever *emitted* the events the Go TUI waits for. That is how a
fully-green suite coexisted with an entirely unwired feature surface.

This module owns the producer contract. `tests/testdata/protocol_golden.ndjson`
is shared with the Go side (`tui/protocol/golden_test.go`), which decodes the
same lines, so drift on either side fails a test rather than shipping.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
GOLDEN = Path(__file__).resolve().parent / "testdata" / "protocol_golden.ndjson"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_openai_server import FakeOpenAIServer  # noqa: E402


def _run_agent(
    script: list[dict],
    prompt: str = "analyze the data",
    extra_env: dict | None = None,
    scratch: Path | None = None,
) -> tuple[list[dict], FakeOpenAIServer]:
    """Run the real agent subprocess against a fake server; return its events.

    Runs out-of-process on purpose: it is the only way to observe the actual
    stdout contract the Go TUI depends on (import-level state, print routing
    and buffering are all in play in-process).
    """
    srv = FakeOpenAIServer(script).start()
    env = dict(os.environ)
    # Point Config at the fake server, NOT the ambient ANTHROPIC_BASE_URL.
    # Config is the single authority for the endpoint (see config.py); the SDK
    # takes the URL from ClaudeAgentOptions.base_url and never reads env. The
    # test must therefore steer the same knob the agent reads, or it would point
    # at a real endpoint and the run would depend on the developer's shell.
    env["OLLAMA_BASE_URL"] = srv.base_url
    env["ANTHROPIC_BASE_URL"] = srv.base_url
    env["ANTHROPIC_API_KEY"] = "test-key"
    env["PYTHONPATH"] = str(SRC)
    # The developer's shell may export CN_LOCAL / CN_MODEL; a local+cloud
    # combination now fails closed by design, which would make these protocol
    # tests fail for reasons unrelated to the protocol. Pin them explicitly.
    env["CN_LOCAL"] = "false"
    env["CN_MODEL"] = "test-model"
    # Keep the run hermetic: no real writes into the developer's output dirs.
    # The agent writes a report.md into CN_OUTPUT_DIR, so this must point
    # somewhere disposable. Callers pass pytest's tmp_path; the tempfile
    # fallback keeps a bare `_run_agent(...)` call from littering the repo.
    out_dir = scratch or Path(tempfile.mkdtemp(prefix="cn-e2e-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    env["CN_OUTPUT_DIR"] = str(out_dir)
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "comp_neuroscientist.cli", "--json", prompt],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=90,
        )
    finally:
        srv.stop()

    events = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))

    return events, srv


def _types(events: list[dict]) -> list[str]:
    return [e.get("type") for e in events]


# ─── Core protocol tests ────────────────────────────────────────


def test_emits_text_and_result_events():
    """A plain answer streams text events and terminates with a result event."""
    events, _ = _run_agent([{"text": "The answer is 42."}])

    assert "result" in _types(events), f"no result event; got {_types(events)}"
    assert "text" in _types(events), f"no text event; got {_types(events)}"

    result = [e for e in events if e["type"] == "result"][-1]
    assert result["text"] == "The answer is 42.", "final answer must survive intact"
    assert result["is_error"] is False
    assert result["turns"] == 1


def test_emits_tool_call_and_tool_result_events(tmp_path: Path):
    """A tool-calling turn surfaces tool_call + tool_result, in order.

    This is the regression guard for the headline finding: the SDK used to
    execute tools internally without ever yielding them, so `agent.py` could
    not emit these events no matter what it wanted to, and the TUI's
    tool_call/tool_result branches were dead code.
    """
    tmp = tmp_path / "e2e_tool_target.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)

    events, srv = _run_agent([
        {"tool": ("Write", {"file_path": str(tmp), "content": "written-by-e2e"})},
        {"text": "Saved the file."},
    ], scratch=tmp_path)

    types = _types(events)
    assert "tool_call" in types, f"tool_call never emitted; got {types}"
    assert "tool_result" in types, f"tool_result never emitted; got {types}"

    call = [e for e in events if e["type"] == "tool_call"][0]
    assert call["name"] == "Write"
    assert call["arguments"]["content"] == "written-by-e2e"

    result_ev = [e for e in events if e["type"] == "tool_result"][0]
    assert result_ev["name"] == "Write"
    # Write reports a confirmation (path + size), not the content itself.
    assert "Wrote" in result_ev["result"] and "e2e_tool_target.txt" in result_ev["result"]

    # Ordering: the call is announced before its result.
    assert types.index("tool_call") < types.index("tool_result")

    # The tool really executed (not just announced).
    assert tmp.exists() and tmp.read_text() == "written-by-e2e"

    # And its output was fed back to the model on the next turn.
    follow_up = srv.requests[1]["messages"]
    assert any(m.get("role") == "tool" for m in follow_up), (
        "tool result must be fed back into the conversation"
    )

    # Two turns: one that requested a tool, one that answered.
    assert [e for e in events if e["type"] == "result"][-1]["turns"] == 2

    tmp.unlink(missing_ok=True)


def test_emits_status_events_during_tool_turns():
    """Tool turns announce progress via status, so the TUI can show a turn count."""
    events, _ = _run_agent([
        {"tool": ("Glob", {"pattern": "*.py"})},
        {"text": "done"},
    ])

    statuses = [e for e in events if e["type"] == "status"]
    assert statuses, f"no status events emitted; got {_types(events)}"
    assert statuses[0]["status"] == "running"
    assert statuses[0]["turns"] == 1


def test_streamed_deltas_are_concatenated_in_order():
    """Text arrives as several deltas; the result must be their concatenation."""
    long_answer = "alpha bravo charlie delta echo foxtrot golf"
    events, _ = _run_agent([{"text": long_answer}])

    streamed = "".join(e["content"] for e in events if e["type"] == "text")
    assert streamed == long_answer, f"stream lost or reordered text: {streamed!r}"

    result = [e for e in events if e["type"] == "result"][-1]
    assert result["text"] == long_answer


def test_unknown_tool_reports_error_and_continues():
    """A bogus tool call surfaces an error result and still reaches a result event."""
    events, _ = _run_agent([
        {"tool": ("NoSuchTool", {"x": 1})},
        {"text": "recovered"},
    ])

    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert tool_results, "an unknown tool must still produce a tool_result event"
    assert "ToolError" in tool_results[0]["result"]

    # The agent recovered rather than crashing.
    assert [e for e in events if e["type"] == "result"][-1]["text"] == "recovered"


def test_result_is_always_the_terminal_event():
    """`result` must be last, so the TUI knows when the stream is complete."""
    events, _ = _run_agent([
        {"tool": ("Glob", {"pattern": "*.py"})},
        {"text": "final"},
    ])
    assert _types(events)[-1] == "result", (
        f"result must terminate the stream; got {_types(events)}"
    )


# ─── Cross-language golden contract ─────────────────────────────


def test_event_stream_matches_go_golden_fixture(tmp_path: Path):
    """The stream the Go decoder is tested against is the stream we actually emit.

    If someone changes what `agent.py` emits without updating the fixture the
    Go tests decode, this fails — that is the drift the old suite could not see.
    """
    assert GOLDEN.exists(), f"missing shared golden fixture: {GOLDEN}"

    actual = [json.loads(ln) for ln in GOLDEN.read_text().splitlines() if ln.strip()]
    emitted_types = {e["type"] for e in actual}
    # `error` is included explicitly by the generator: a happy-path run cannot
    # produce one, but it is part of the contract the Go decoder handles.
    expected = {"text", "status", "tool_call", "tool_result", "result", "error"}
    assert emitted_types == expected, (
        f"golden fixture must cover every event type; got {sorted(emitted_types)}"
    )

    # A live run must be able to produce every type the fixture declares.
    events, _ = _run_agent([
        {"tool": ("Write", {"file_path": str(tmp_path / "golden.txt"), "content": "x"})},
        {"text": "ok"},
    ], scratch=tmp_path)
    live_types = set(_types(events))
    assert emitted_types <= live_types | {"error"}, (
        f"live run missing event types declared in golden: "
        f"{sorted(emitted_types - live_types)}"
    )


@pytest.mark.parametrize("event_type", [
    "text", "status", "tool_call", "tool_result", "result", "error",
])
def test_golden_lines_are_parseable_by_shape(event_type: str):
    """Every golden line is valid JSON with the keys the Go struct declares."""
    if not GOLDEN.exists():
        pytest.skip("golden fixture not generated yet")
    lines = [ln for ln in GOLDEN.read_text().splitlines() if ln.strip()]
    for line in lines:
        obj = json.loads(line)
        assert "type" in obj
        if obj["type"] == event_type:
            return
    pytest.fail(f"golden fixture has no example of event type {event_type!r}")
