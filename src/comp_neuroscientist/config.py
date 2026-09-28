"""
Configuration — Ollama model, paths, env vars.

Config is the SINGLE AUTHORITY for where requests go. The SDK reads the
endpoint from `ClaudeAgentOptions.base_url`, never from ambient environment
variables, so `--local` cannot be silently defeated by a pre-existing
`ANTHROPIC_BASE_URL` in the caller's shell.
"""

import os
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

# Default model: local models for privacy, cloud models for power.
# Set CN_MODEL to override, or use --model flag.
_DEFAULT_MODEL = "llama3.1"
_DEFAULT_CLOUD_MODEL = "deepseek-v4-flash:cloud"

# Hosts that count as "this machine". Anything else means data leaves the host.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1"})


class Config(BaseModel):
    """Comp-Neuroscientist agent configuration."""

    model_config = {"validate_assignment": True}

    model: str = Field(
        default_factory=lambda: _resolve_default_model()
    )

    ollama_url: str = Field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    max_turns: int = Field(
        default_factory=lambda: _safe_int_env("CN_MAX_TURNS", 30)
    )
    effort: str = Field(
        default_factory=lambda: os.environ.get("CN_EFFORT", "high")
    )

    output_dir: str = Field(
        default_factory=lambda: os.environ.get("CN_OUTPUT_DIR", "results")
    )

    local: bool = Field(
        default_factory=lambda: os.environ.get("CN_LOCAL", "").lower() in ("1", "true", "yes")
    )

    @model_validator(mode="after")
    def _enforce_local_privacy(self) -> "Config":
        """Fail closed: local mode may not be routed off-host.

        `local` is a privacy guarantee ("no data leaves your machine"), so it is
        treated as a constraint to enforce, not a hint to pass along. Two ways
        the promise could previously be broken and neither was caught:

        1. The endpoint came from `ANTHROPIC_BASE_URL` via `setdefault`, which by
           definition does not override a value the caller already exported. A
           user who followed the README's `export ANTHROPIC_BASE_URL=...` and then
           passed `--local` sent data to a remote host while the tool reported
           local mode.
        2. A `:cloud`-suffixed model routed to a remote gateway even on a local
           endpoint, leaking the prompt in the request body.

        Both now raise rather than degrade silently. Silently routing sensitive
        data outward while claiming local is worse than refusing to start.
        """
        if not self.local:
            return self
        host = _url_host(self.ollama_url)
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(
                f"--local requested but OLLAMA_BASE_URL points at {host!r}, "
                f"which is not a loopback address. Local mode is a promise that "
                f"no data leaves this machine, so it refuses to run rather than "
                f"send data to a remote host. Unset OLLAMA_BASE_URL to use a "
                f"local Ollama, or drop --local to use a cloud model."
            )
        if _model_is_cloud(self.model):
            raise ValueError(
                f"--local requested but model {self.model!r} is a cloud model. "
                f"Cloud models are served off-machine regardless of the endpoint, "
                f"so the request body still leaves this host. Choose a local model "
                f"(e.g. 'llama3.1') or drop --local."
            )
        return self

    @property
    def resolved_base_url(self) -> str:
        """The endpoint the SDK will actually call, normalized to a /v1 path.

        This is the value the agent uses verbatim. It is deliberately a plain
        property rather than a cached field: the SDK must not be able to read a
        different endpoint from anywhere else.
        """
        base = self.ollama_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @property
    def api_key(self) -> str:
        """Credential for the resolved endpoint.

        Local Ollama ignores the value but the OpenAI client requires a
        non-empty one, so a placeholder is supplied when no token is set.

        In local mode the placeholder is returned *unconditionally*. Local mode
        is a promise that nothing sensitive reaches a cloud service, and a
        `sk-ant-...` token read from the ambient environment would be handed to
        the local client as its credential. It is not sent off-host (the
        endpoint is loopback, enforced by `_enforce_local_privacy`), but a cloud
        key has no business being attached to a local run at all: it widens the
        blast radius if the endpoint is ever misrouted, and it can surface in
        logs, crash dumps, and error messages. Local mode gets a local key.
        """
        if self.local:
            return "ollama"
        return os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get(
            "ANTHROPIC_AUTH_TOKEN", ""
        ) or "ollama"

    @property
    def plots_dir(self) -> str:
        return f"{self.output_dir}/plots"

    @property
    def models_dir(self) -> str:
        return f"{self.output_dir}/models"

    @property
    def processed_dir(self) -> str:
        return f"{self.output_dir}/processed"

    @classmethod
    def from_env(cls) -> "Config":
        return cls()

    def model_dump(self, **kwargs) -> dict:
        return super().model_dump(**kwargs)


def _url_host(url: str) -> str:
    """Return the lowercased hostname of a URL, or "" if it cannot be parsed.

    Deliberately fails toward an empty string: an unparseable endpoint is not
    loopback, so the caller's privacy check rejects it.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    return (parsed.hostname or "").lower()


def _model_is_cloud(model: str) -> bool:
    """True when a model name denotes a cloud-served model.

    Ollama's convention is a `:cloud` tag suffix (e.g. `deepseek-v4-flash:cloud`).
    The check is a suffix test on the tag only, so a model merely *named*
    "cloud-foo" is not misread as remote.
    """
    tag = model.rsplit(":", 1)[-1] if ":" in model else ""
    return tag.lower() == "cloud"


def _resolve_default_model() -> str:
    """Return local or cloud default based on CN_LOCAL env var."""
    env_model = os.environ.get("CN_MODEL", "")
    if env_model:
        return env_model
    if os.environ.get("CN_LOCAL", "").lower() in ("1", "true", "yes"):
        return _DEFAULT_MODEL
    return _DEFAULT_CLOUD_MODEL


def _safe_int_env(key: str, default: int, minimum: int = 1) -> int:
    """Read an integer env var, falling back to `default` when unusable.

    Fails closed: a value that is not an integer, or that falls below
    `minimum`, yields `default` instead of propagating a nonsensical setting
    (e.g. CN_MAX_TURNS=-1, which would mean "unlimited" to a caller that
    never checks for it).
    """
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < minimum:
        return default
    return value


config = Config.from_env()
