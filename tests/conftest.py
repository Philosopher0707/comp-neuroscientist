"""Fixtures for comp-neuroscientist tests."""
import sys
from pathlib import Path

# Ensure src is on the path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
