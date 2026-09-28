"""Pin the real containment behaviour of _safe_resolve.

D1: the old docstring claimed _safe_resolve kept paths "inside the working
directory tree". It never did -- resolve() normalizes, it does not confine.
This test witnesses the ACTUAL behaviour, so a future reader who wants a real
boundary must add one deliberately (and this test will fail loudly when they
change the contract) rather than trusting a docstring that was lying.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claude_agent_sdk import _safe_resolve  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A cwd containing a tree plus an escape target outside it."""
    (tmp_path / "inside" / "nested").mkdir(parents=True)
    (tmp_path / "inside" / "file.txt").write_text("data")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_relative_path_is_anchored_at_cwd(sandbox):
    assert _safe_resolve("inside/file.txt") == sandbox / "inside" / "file.txt"


def test_dotdot_traversal_is_NOT_confined(sandbox):
    """Documents the gap D1 described: .. escapes cwd rather than being blocked."""
    resolved = _safe_resolve("../outside.txt")
    assert resolved == sandbox.parent / "outside.txt"
    assert sandbox not in resolved.parents


def test_absolute_path_is_NOT_confined(sandbox, tmp_path):
    """An absolute path is honoured verbatim -- also outside cwd."""
    target = tmp_path / "outside.txt"
    assert _safe_resolve(str(target)) == target


def test_tilde_is_expanded(sandbox):
    assert _safe_resolve("~").is_absolute()
    assert _safe_resolve("~") == Path.home().resolve()


def test_symlinks_are_resolved(sandbox):
    (sandbox / "link.txt").symlink_to(sandbox / "inside" / "file.txt")
    assert _safe_resolve("link.txt") == (sandbox / "inside" / "file.txt").resolve()
