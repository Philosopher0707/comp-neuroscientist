"""Tests for Bash command safety.

Regression witness for the inverted denylist: the old pattern
`\\brm\\s+-rf\\s*/\\b` placed \\b after '/', a non-word char, so it required a
word char next — blocking `rm -rf /Users/x` while ALLOWING `rm -rf /`.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claude_agent_sdk import _deny_reason  # noqa: E402


# ─── the canonical catastrophic command ────────────────────────
# Built by concatenation so this file's text is not itself a literal attack string.
RM = "rm"
RF = "-rf /"


@pytest.mark.parametrize("cmd", [
    RM + " " + RF,                    # bare root
    RM + " -fr /",                    # swapped flags
    RM + " -r -f /",                  # split flags
    RM + " --recursive --force /",    # long flags
    RM + " -rf /*",                   # root glob
    RM + " -rf ~",                    # home
    RM + " -rf $HOME",                # env var
    RM + " -rf .",                    # cwd
    RM + " -rf /Users/philosopher",   # the one the OLD denylist caught
    RM + " -rf /etc",
    RM + " -rf /usr/lib",
    "rmdir /",
    "shred -u /etc/hosts",
    "chmod -R 777 /",
    "chown -R me /",
    "truncate -s 0 /dev/sda",
])
def test_destructive_system_targets_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── shell expansion / chaining bypasses ────────────────────────
@pytest.mark.parametrize("cmd", [
    "ls && " + RM + " " + RF,        # chain
    "true; " + RM + " " + RF,        # sequence
    "ls | tee /tmp/x",              # pipe
    "echo `id`",                    # backtick substitution
    "echo $(whoami)",               # $() substitution
])
def test_metacharacters_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── destruction expressed in another language ──────────────────
@pytest.mark.parametrize("cmd", [
    'python -c "import shutil; shutil.rmtree(\'/\')"',
    "python -c 'import os; os.remove(\"/etc/hosts\")'",
    "perl -e 'unlink for glob \"/*\"'",
    "find / -delete",
    "find / -exec rm {} +",
    "git clean -fdx",
])
def test_non_regex_destructive_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── outside the workspace, whatever the command ────────────────
@pytest.mark.parametrize("cmd", [
    RM + " -rf /etc/hosts",
    RM + " -rf ../sibling",
    RM + " -rf /tmp/other_project",
    "chmod 777 /etc/hosts",
    "truncate -s 0 /var/log/system.log",
])
def test_destructive_outside_workspace_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── the product must still work ────────────────────────────────
# The agent legitimately runs Python/R for analysis and writes report.md via
# redirection; over-blocking breaks it. These are the anti-regression guard.
@pytest.mark.parametrize("cmd", [
    "ls -la",
    "python -c 'import numpy; print(numpy.__version__)'",
    "python3 -c 'print(1)'",
    "python -c 'import os; print(os.getcwd() and 1)'",
    "grep -rn foo .",
    "cat README.md",
    "wc -l src/claude_agent_sdk.py",
    RM + " build/out.txt",            # ordinary, inside workspace
    RM + " -rf build",                # ordinary dir, inside workspace
    "chmod +x scripts/run.sh",
    "cat /etc/passwd > /tmp/x",      # redirection is legitimate
    "python -c 'import numpy as np; a = np.zeros(3); print(a.mean())'",
    # `;` inside a string literal is a Python statement separator, not chaining.
    "python -c 'print(\"a;b\")'",
    # Banned text inside a string is text, not a command.
    "python -c 'print(\"rm -rf /\")'",
    # `make` is only denied for its deleting TARGETS, not wholesale.
    "make build",
    "make -j8 all",
    "make clean-build",             # prefix of a banned target
    "make cleanup_data.py",         # prefix, with a suffix
])
def test_legitimate_commands_allowed(cmd):
    assert _deny_reason(cmd) is None, f"OVER-BLOCKED: {cmd}"


# ─── deleting make targets ─────────────────────────────────────
# `make` is legitimate, so only the targets that wipe build output are denied.
@pytest.mark.parametrize("cmd", [
    "make clean",
    "make distclean",
    "make realclean",
])
def test_destructive_make_targets_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── xargs ─────────────────────────────────────────────────────
# xargs APPENDS stdin arguments to a command, so unwrapping one token cannot
# model it. It has no legitimate use here, so it is denied outright.
@pytest.mark.parametrize("cmd", [
    "xargs rm",
    "xargs -0 rm",
    "xargs rm < /tmp/list",
])
def test_xargs_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── wrapper commands take their OWN options first ──────────────
# The payload is the first NON-OPTION token, not simply the next token.
# "env -i rm -rf /" must be judged as "rm -rf /", and "sudo -u root" must skip
# both the flag and its value.
@pytest.mark.parametrize("cmd", [
    "env -i " + RM + " /",
    "sudo -u root " + RM + " /",
    "sudo --user=root " + RM + " /",
    "nohup rm -rf /",
    "command rm -rf /",
    "time rm -rf /",
])
def test_wrapper_option_skipping(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── shell -c payloads are re-judged, not pattern-scanned ────────
@pytest.mark.parametrize("cmd", [
    "sh -c 'rm -rf /'",
    "bash -c 'rm -rf /'",
    "sh -c 'rm -rf ~'",
    "bash -c 'rm -rf $HOME'",
])
def test_shell_c_payload_rejudged(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# ─── destruction in other languages ─────────────────────────────
@pytest.mark.parametrize("cmd", [
    "ruby -e 'FileUtils.rm_rf(\"x\")'",
    "node -e 'require(\"fs\").rmSync(\"x\")'",
    "perl -e 'unlink for glob(\"*\")'",
    "python -c 'import shutil; shutil.rmtree(\"build\")'",
    "python -c 'import os; os.remove(\"x\")'",
])
def test_scripted_destruction_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"
