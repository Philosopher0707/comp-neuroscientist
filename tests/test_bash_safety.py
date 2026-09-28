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


# ─── regression: holes found by adversarial probing ─────────────
# Each block below is a bypass that the policy previously allowed. They are
# kept as witnesses so the same spelling cannot silently come back.

# 1. The `-c` flag clusters with other options. The policy matched
#    `parts[1] == "-c"` exactly, so `sh -ec '<payload>'` skipped the recursion
#    that `sh -c '<payload>'` got. `-e` takes no value, so the scan must
#    continue past it rather than treat the next token as the code.
@pytest.mark.parametrize("cmd", [
    "sh -ec 'rm -rf /'",
    "sh -e -c 'rm -rf /'",
    "sh -xc 'rm -rf /'",
    "sh --command 'rm -rf /'",
    "bash -ec 'rm -rf /'",
    "zsh -ec 'rm -rf ~'",
    "sh -c \"sh -ec 'rm -rf /'\"",
])
def test_shell_flag_cluster_payload_rejudged(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


@pytest.mark.parametrize("cmd", [
    "sh -ec 'echo hi'",
    "sh -c 'echo hi'",
])
def test_shell_flag_cluster_harmless_allowed(cmd):
    assert _deny_reason(cmd) is None, f"OVERBLOCKED: {cmd}"


# 2. Redirect targets. The policy comment claimed the target was
#    containment-checked, but no such check existed: `echo x > /etc/hosts` was
#    allowed. It also missed a file-descriptor prefix, so `2> /etc/hosts` slipped
#    past the `>` scan.
@pytest.mark.parametrize("cmd", [
    "echo pwned > /etc/hosts",
    "echo x > /etc/sudoers",
    "echo x > /bin/sh",
    "echo x >> /etc/hosts",
    "ls 2> /etc/hosts",
    "echo x > /etc/hosts.bak",
    "echo x > /usr/lib/evil.so",
])
def test_redirect_into_protected_target_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# Persistence: home dirs are excluded from _SYSTEM_TARGETS because the
# workspace usually lives in one, so a write there must still be refused.
@pytest.mark.parametrize("cmd", [
    "echo x > ~/.ssh/authorized_keys",
    "echo x >> ~/.ssh/authorized_keys",
    "echo x > ~/.zshrc",
    "echo x >> ~/.bashrc",
    "echo x > ~/.aws/credentials",
    "echo x > ~/.netrc",
    "cp payload ~/.ssh/authorized_keys",
])
def test_persistence_write_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# Anti-regression for the fix above: reading those files stays legitimate, and
# so does writing a scratch file, which the allow-list already pins.
@pytest.mark.parametrize("cmd", [
    "cat ~/.gitconfig",
    "cat ~/.ssh/id_rsa",
    "wc -l < /etc/passwd",
    "cat /etc/passwd > /tmp/x",
    "echo x > out.txt",
    "python3 -c 'print(1 > 0)'",
    "python3 -c 'print('a > b')'",
])
def test_reads_and_scratch_writes_still_allowed(cmd):
    assert _deny_reason(cmd) is None, f"OVERBLOCKED: {cmd}"


# 3. The write side. Only deletion/metadata were boundary-checked, so copying
#    a file outside the workspace was free: `cp payload /etc/launchd/x.plist`.
@pytest.mark.parametrize("cmd", [
    "cp payload /etc/launchd/evil.plist",
    "cp -r src /etc/x",
    "install -m755 payload /usr/local/bin/x",
    "rsync -a src/ /etc/x",
    "ln -s payload /bin/evil",
])
def test_copy_into_protected_target_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


@pytest.mark.parametrize("cmd", [
    "cp a b",
    "cp a /tmp/b",
    "ln -s /etc/shadow ./s",
])
def test_copies_within_allowed_area_still_allowed(cmd):
    assert _deny_reason(cmd) is None, f"OVERBLOCKED: {cmd}"


# 4. Interpreter payload spellings the pattern list missed: a bare call after
#    `from os import remove`, an aliased receiver (`import os as o`), and
#    subprocess entry points that shell out.
@pytest.mark.parametrize("cmd", [
    "python3 -c 'from os import remove; remove(\"/etc/hosts\")'",
    "python3 -c 'import os as o; o.remove(\"/etc/hosts\")'",
    "python3 -c 'import subprocess; subprocess.run(\"rm -rf /\", shell=True)'",
    "python3 -c 'import subprocess as s; s.check_output(\"rm -rf /\", shell=True)'",
    "python3 -c 'import os; os.system(\"rm -rf /\")'",
    "python3 -c 'import shutil; shutil.move(\"/etc\", \"/tmp\")'",
    "python3 -c 'rmdir(\"/etc/nginx\")'",
    "ruby -e 'File.delete(\"/etc/hosts\")'",
    "node -e 'require(\"fs\").unlinkSync(\"/etc/hosts\")'",
])
def test_interpreter_destruction_spellings_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# Anti-regression: the added patterns must not catch ordinary analysis code.
@pytest.mark.parametrize("cmd", [
    "python -c 'import numpy; print(numpy.__version__)'",
    "python -c 'import os; print(os.getcwd() and 1)'",
    "python3 -c 'print([x.remove if hasattr(x, \"remove\") else 0 for x in []])'",
    "python3 -c 'print(1)'",
])
def test_ordinary_inline_code_still_allowed(cmd):
    assert _deny_reason(cmd) is None, f"OVERBLOCKED: {cmd}"


# 5. Autostart persistence. The write-side home rules covered ~/.ssh and the
#    shell rc files but not the directories the OS actually re-runs at login,
#    so `> ~/Library/LaunchAgents/evil.plist` was a free persistence install.
@pytest.mark.parametrize("cmd", [
    "echo x > ~/Library/LaunchAgents/evil.plist",
    "echo x >> ~/Library/LaunchAgents/evil.plist",
    "echo x > ~/Library/StartupItems/evil.plist",
    "echo x > ~/.config/autostart.desktop",
    "echo x > ~/.config/autostart.d/evil.desktop",
    "echo x > ~/.config/systemd/user/evil.service",
    "cp payload ~/.emacs.d/init.el",
    "tee ~/.ssh/authorized_keys",
])
def test_autostart_persistence_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


# 6. `tee`. It writes every file operand, which is the same write as `> path`
#    with no redirect token present. It was in no command set at all: the piped
#    form only got denied by the unrelated "no `|`" rule, leaving the bare
#    `tee /etc/hosts` writable. Last-operand-only checking also missed
#    `tee out.log /etc/hosts`, where the protected path is not last.
@pytest.mark.parametrize("cmd", [
    "tee /etc/hosts",
    "tee -a /bin/sh",
    "tee /etc/sudoers",
    "tee out.log /etc/hosts",
    "tee /etc/hosts out.log",
])
def test_tee_into_protected_target_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


@pytest.mark.parametrize("cmd", [
    "tee out.log",
    "tee -a /tmp/run.log",
])
def test_tee_scratch_still_allowed(cmd):
    assert _deny_reason(cmd) is None, f"OVERBLOCKED: {cmd}"


# 7. `mv` writes to its SOURCE as well as its destination: the original path
#    stops existing. Judging only the destination left `mv ~/Library/Keychains k`
#    free, which moves the user's keychain out from under the system.
@pytest.mark.parametrize("cmd", [
    "mv ~/Library/Keychains ./k",
    "mv ~/Library/Keychains/chat.key ./x",
    "mv ~/.ssh/id_rsa /tmp/x",
])
def test_mv_out_of_protected_target_denied(cmd):
    assert _deny_reason(cmd) is not None, f"NOT DENIED: {cmd}"


@pytest.mark.parametrize("cmd", [
    "mv a b",
    "mv build/out.txt dist/",
    "mv a /tmp/b",
])
def test_moves_within_policy_still_allowed(cmd):
    assert _deny_reason(cmd) is None, f"OVERBLOCKED: {cmd}"


# Deleting an autostart directory is refused, and that is correct: it is a
# protected target, and `rm` is always containment-checked. Pinned so the
# protection added above cannot be mistaken for an over-block.
def test_rm_of_autostart_dir_denied():
    assert _deny_reason("rm -rf ~/Library/LaunchAgents") is not None
