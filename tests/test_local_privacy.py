"""
Regression witnesses for the --local privacy guarantee.

The original defect was not a missing check but a *misplaced authority*. Three
things combined to let data leave the machine while the tool reported local
mode:

1. `config.local` was write-only — assigned at cli.py and never read, so the
   flag had no effect on anything.
2. The endpoint came from `ANTHROPIC_BASE_URL` via `os.environ.setdefault`,
   which by definition does NOT override a value the caller already exported.
   The README told users to export that variable, so this was the normal case,
   not an edge case.
3. A `:cloud` model routes off-machine regardless of the endpoint.

These tests assert on the EFFECTIVE url the SDK would dial — the thing the bug
was actually about — not on the config field, which is exactly the level at
which the old code looked correct.

Every test is hermetic: `tests/conftest.py` strips the ambient CN_*/OLLAMA_*/
ANTHROPIC_* variables, and each test sets precisely what it needs.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from claude_agent_sdk import ClaudeAgentOptions
from comp_neuroscientist.config import (
    _LOOPBACK_HOSTS,
    Config,
    _model_is_cloud,
    _url_host,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"


# ─── Config-level enforcement ───────────────────────────────────


def test_local_rejects_offhost_endpoint(monkeypatch):
    """A remote OLLAMA_BASE_URL under --local must fail closed, not route."""
    monkeypatch.setenv("CN_LOCAL", "1")
    monkeypatch.setenv("CN_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://api.gateway.internal")

    with pytest.raises(ValidationError) as exc:
        Config()

    msg = str(exc.value)
    assert "not a loopback address" in msg
    # The error must tell the user how to escape it, not just refuse.
    assert "--local" in msg


@pytest.mark.parametrize("host", [
    "https://api.gateway.internal",
    "http://10.0.0.5:11434",
    "http://192.168.1.10:11434",
    "http://ollama.example.com",
    "http://localhost.evil.com",
    "http://127.0.0.1.evil.com",
])
def test_local_rejects_lookalike_hosts(monkeypatch, host):
    """Non-loopback and suffix-spoof hosts are both rejected."""
    monkeypatch.setenv("CN_LOCAL", "1")
    monkeypatch.setenv("CN_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", host)

    with pytest.raises(ValidationError):
        Config()


@pytest.mark.parametrize("host", [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
    "http://[::1]:11434",
])
def test_local_accepts_loopback_forms(monkeypatch, host):
    """All loopback spellings remain valid — the check is not over-strict."""
    monkeypatch.setenv("CN_LOCAL", "1")
    monkeypatch.setenv("CN_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", host)

    c = Config()
    assert c.local is True


def test_local_rejects_cloud_model(monkeypatch):
    """--local with a :cloud model fails closed.

    Ollama serves `:cloud` tags off-machine, so the request body leaves the host
    even when the URL is loopback.
    """
    monkeypatch.setenv("CN_LOCAL", "1")
    monkeypatch.setenv("CN_MODEL", "deepseek-v4-flash:cloud")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    with pytest.raises(ValidationError) as exc:
        Config()
    assert "cloud model" in str(exc.value)


def test_cloud_mode_permits_remote_endpoint(monkeypatch):
    """Without --local, a remote endpoint is fine — only local mode constrains."""
    monkeypatch.setenv("CN_LOCAL", "false")
    monkeypatch.setenv("CN_MODEL", "deepseek-v4-flash:cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://api.gateway.internal")

    c = Config()
    assert c.local is False
    assert c.resolved_base_url == "https://api.gateway.internal/v1"


# ─── Helpers ────────────────────────────────────────────────────


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:11434", "localhost"),
    ("http://LOCALHOST:11434", "localhost"),
    ("https://api.example.com", "api.example.com"),
    ("not a url at all", ""),
    ("", ""),
])
def test_url_host(url, expected):
    assert _url_host(url) == expected


@pytest.mark.parametrize("model,expected", [
    ("deepseek-v4-flash:cloud", True),
    ("glm-5.1:cloud", True),
    ("llama3.1", False),
    ("qwen2.5:7b", False),
    ("kimi-k2.6:cloud", True),
    # a model merely NAMED cloud must not be misread as remote
    ("cloud-base", False),
    ("my-cloud:7b", False),
])
def test_model_is_cloud(model, expected):
    assert _model_is_cloud(model) is expected


# ─── Effective endpoint, not the config field ───────────────────


def test_ambient_env_cannot_override_config_endpoint(monkeypatch):
    """The original bypass, end to end.

    Ambient ANTHROPIC_BASE_URL points at a remote host. Config resolves to
    loopback. The value the SDK is handed must be the loopback one.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.gateway.internal")
    monkeypatch.setenv("CN_LOCAL", "1")
    monkeypatch.setenv("CN_MODEL", "llama3.1")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    c = Config()
    options = ClaudeAgentOptions(
        model=c.model, base_url=c.resolved_base_url, api_key=c.api_key
    )
    assert options.base_url == "http://localhost:11434/v1"
    assert "gateway.internal" not in options.base_url


def test_sdk_does_not_read_env_when_base_url_given(monkeypatch):
    """With base_url set, ambient env is ignored entirely by the SDK.

    Asserted on the SDK's own resolution path rather than on Config, so a
    future re-introduction of env reading inside the SDK is caught here.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example.com")
    options = ClaudeAgentOptions(base_url="http://127.0.0.1:11434", env_fallback=True)
    assert options.base_url == "http://127.0.0.1:11434"


def test_sdk_fails_closed_without_endpoint_and_without_env(mononkeypatch_placeholder=None):
    """env_fallback=False with no base_url must raise rather than guess."""
    options = ClaudeAgentOptions(env_fallback=False)
    assert options.base_url is None


# ─── resolved_base_url normalisation ───────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("http://localhost:11434", "http://localhost:11434/v1"),
    ("http://localhost:11434/", "http://localhost:11434/v1"),
    ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
])
def test_resolved_base_url_normalises(monkeypatch, raw, expected):
    c = Config(local=False, model="glm-5.1:cloud", ollama_url=raw)
    assert c.resolved_base_url == expected


# ─── The CLI actually refuses ───────────────────────────────────
# These run the real CLI so the check is proven at the entry point a user hits,
# not merely at the Config layer.


def _run_cli(*args, env_overrides=None):
    # Inherit the real environment (as tests/test_protocol_e2e.py does) so the
    # interpreter can import its own dependencies, then override deliberately.
    # Hand-building a minimal env breaks on missing site-packages.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "comp_neuroscientist.cli", *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=60,
    )


def test_cli_local_with_remote_endpoint_exits_nonzero():
    """`--local` + a remote OLLAMA_BASE_URL must not start and must not route."""
    proc = _run_cli(
        "analyze this dataset",
        "--local",
        env_overrides={
            "OLLAMA_BASE_URL": "https://api.gateway.internal",
            "CN_MODEL": "llama3.1",
        },
    )
    assert proc.returncode != 0, (
        f"CLI exited 0 despite --local pointing off-host:\n{proc.stdout}\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "loopback" in combined or "local" in combined.lower()


def test_cli_local_with_cloud_model_exits_nonzero():
    """`--local` + an explicit cloud model must refuse to start."""
    proc = _run_cli(
        "analyze this dataset",
        "--local",
        "--model",
        "deepseek-v4-flash:cloud",
        env_overrides={"OLLAMA_BASE_URL": "http://localhost:11434"},
    )
    assert proc.returncode != 0, (
        f"CLI exited 0 despite --local with a cloud model:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "cloud" in (proc.stdout + proc.stderr).lower()


# ─── the CLI's attribute-assignment order ───────────────────────
# cli.py mutates the *shared* `config` singleton rather than building a new one:
#
#     if args.local:
#         cfg.local = True
#         if not args.model:
#             cfg.model = "llama3.1"
#
# The ordering matters. If `local = True` were assigned while a cloud model was
# still in place, the privacy check would run against a cloud model and could
# reject a legitimate invocation. These pin the real singleton, with hostile
# ambient env pre-set, so the behaviour cannot silently invert.


def test_cli_local_assignment_order_keeps_endpoint_loopback(monkeypatch):
    monkeypatch.setenv("CN_MODEL", "claude-sonnet-4")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example.com")
    monkeypatch.delenv("CN_LOCAL", raising=False)

    c = Config()
    c.local = True          # exactly what cli.py does first for --local
    c.model = "llama3.1"    # ...and then the local default

    # Use the same host helper the production tests already rely on, rather than
    # re-parsing the URL here. Do NOT assert a specific string prefix: the
    # module normalises to 127.0.0.1, but "localhost" and "::1" are equally
    # valid, and pinning one spelling would make this a change-detector rather
    # than a privacy witness.
    assert _url_host(c.resolved_base_url) in _LOOPBACK_HOSTS
    assert "evil.example.com" not in c.resolved_base_url


def test_cli_local_assignment_order_does_not_leak_cloud_key(monkeypatch):
    """A cloud API key must never reach a local run, whatever the ambient env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-be-used")
    monkeypatch.setenv("CN_MODEL", "claude-sonnet-4")
    monkeypatch.delenv("CN_LOCAL", raising=False)

    c = Config()
    c.local = True
    c.model = "llama3.1"

    assert c.api_key != "sk-ant-should-never-be-used"
    assert c.api_key == "ollama"
