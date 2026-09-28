"""
Configuration — Ollama model, paths, env vars.
"""

import os

from pydantic import BaseModel, Field

# Default model: local models for privacy, cloud models for power.
# Set CN_MODEL to override, or use --model flag.
_DEFAULT_MODEL = "llama3.1"
_DEFAULT_CLOUD_MODEL = "deepseek-v4-flash:cloud"


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


def _resolve_default_model() -> str:
    """Return local or cloud default based on CN_LOCAL env var."""
    env_model = os.environ.get("CN_MODEL", "")
    if env_model:
        return env_model
    if os.environ.get("CN_LOCAL", "").lower() in ("1", "true", "yes"):
        return _DEFAULT_MODEL
    return _DEFAULT_CLOUD_MODEL


def _safe_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


config = Config.from_env()
