"""Fixtures for comp-neuroscientist tests."""
import os
import sys
from pathlib import Path

# Ensure src is on the path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Make the suite hermetic w.r.t. the developer's shell.
#
# Config reads CN_* and OLLAMA_* from the environment. Without this, results
# depend on whatever the developer happens to have exported — and a config that
# fails closed (as --local now does) turns an ambient CN_LOCAL/CN_MODEL
# combination into a spurious test failure that has nothing to do with the
# behaviour under test. Individual tests still override these with monkeypatch.
for _var in (
    "CN_MODEL",
    "CN_LOCAL",
    "CN_OUTPUT_DIR",
    "CN_MAX_TURNS",
    "CN_EFFORT",
    "OLLAMA_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
):
    os.environ.pop(_var, None)
