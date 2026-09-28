"""
Production agent loop — talks to Ollama via OpenAI SDK with multi-turn tool execution.

Replaces the single-shot shim with a real multi-turn agent that can:
  • Stream text responses (AssistantMessage chunks)
  • Execute tools (Bash, Read, Write, Glob, Grep)
  • Feed tool results back into the conversation
  • Track turn count and bail at max_turns
  • Switch system prompt when "handing off" to subagents

The public API (query, ClaudeAgentOptions, AgentDefinition, message types) stays
exactly the same so CLI and TUI need no changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional


# ─── Public SDK types (unchanged contract) ───────────────────────

@dataclass
class TextBlock:
    text: str = ""


@dataclass
class AssistantMessage:
    content: List[Any] = field(default_factory=list)
    model: str = ""


@dataclass
class ResultMessage:
    subtype: str = ""
    duration_api_ms: int = 0
    result: str = ""
    num_turns: int = 0
    duration_ms: int = 0
    is_error: bool = False
    errors: str = ""
    session_id: str = ""


@dataclass
class ToolUseMessage:
    """Emitted just before a tool handler runs.

    Additive: consumers that only match AssistantMessage/ResultMessage are
    unaffected, which preserves the documented "public API stays the same"
    contract while making tool activity observable to the event stream.
    """
    name: str = ""
    arguments: dict = field(default_factory=dict)
    tool_use_id: str = ""
    turn: int = 0


@dataclass
class ToolResultMessage:
    """Emitted after a tool handler returns, with its (possibly truncated) result."""
    name: str = ""
    result: str = ""
    tool_use_id: str = ""
    turn: int = 0
    is_error: bool = False


@dataclass
class ClaudeAgentOptions:
    max_turns: int = 30
    model: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    agents: dict = field(default_factory=dict)
    system_prompt: Optional[str] = None


@dataclass
class AgentDefinition:
    description: str = ""
    prompt: str = ""
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None


class AgentLoopError(Exception):
    pass


# ─── Tool schemas (OpenAI function-calling format) ────────────

_TOOL_SCHEMAS: dict[str, dict] = {
    "Bash": {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": (
                "Run a bash shell command and capture its stdout/stderr. "
                "Use for: python scripts, file operations (ls, cat, head), "
                "installing packages, generating plots with matplotlib. "
                "Avoid: interactive commands, deletion of unknown paths, "
                "network access beyond localhost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute. One-liner preferred.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    "Read": {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the full text content of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or relative file path.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    "Write": {
        "type": "function",
        "function": {
            "name": "Write",
            "description": (
                "Write (or overwrite) text content to a file. "
                "Creates parent directories automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Target path."},
                    "content": {"type": "string", "description": "Full content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    "Glob": {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "List files matching a glob pattern (e.g. *.csv, outputs/**/*.png).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern. Double-quoted if passed via shell.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    "Grep": {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search for a keyword or regex pattern across files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Keyword or regex."},
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in.",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
}


def _build_tools(allowed: list[str]) -> list[dict] | None:
    schemas = []
    for name in allowed:
        if name in _TOOL_SCHEMAS:
            schemas.append(_TOOL_SCHEMAS[name])
    return schemas or None


# ─── Safe tool execution ────────────────────────────────────────

_BASH_TIMEOUT = 60          # seconds
_TOOL_TIMEOUT = 10          # seconds for Read/Write/Glob/Grep
_MAX_OUTPUT = 8192          # chars

# Shell metacharacters that enable chaining and substitution.
# `&`, `|`, `;`, backtick and `$(` all start a SECOND command; one Bash call
# is one command. Redirection (>, <) is permitted — the agent legitimately
# writes files that way — but the redirect TARGET is containment-checked below.
_SHELL_CHAIN_CHARS = ("&", "|", ";", "`", "$", "\n")

# Commands that can destroy data or alter system state. Each target argument
# is checked against the workspace boundary rather than pattern-matched.
_DESTRUCTIVE_CMDS = frozenset({
    "rm", "rmdir", "shred", "mkfs", "mkfs.ext4", "fdisk", "diskutil",
    "dd", "truncate", "chmod", "chown", "chgrp", "unlink",
})

# Absolute paths that are never a legal target regardless of the workspace.
# Home directories (/Users, /home, /root) are deliberately NOT listed: the
# workspace usually lives inside one, and listing it would deny the agent's
# own project directory. The boundary rule in _is_inside_workspace covers them.
_SYSTEM_TARGETS = (
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/opt", "/private/etc",
    "/proc", "/sbin", "/System", "/usr", "/var/db", "/var/log", "/var/root",
    "/Applications", "/Library", "/Volumes",
)

# Inline interpreter code (-c / -e) can express any destruction the list above
# blocks. These catch the common forms without parsing a whole language.
_INLINE_DESTRUCTION = [
    r"\brmtree\b",
    r"\bunlink\b",
    r"\bsend2trash\b",
    r"\bos\s*\.\s*remove\s*\(",
    r"\bos\s*\.\s*rmdir\s*\(",
    r"\bos\s*\.\s*removedirs\s*\(",
    r"\bsystem\s*\(\s*['\"]\s*rm\b",
    r"\bpopen\s*\(",
    r"\btruncate\s*\(\s*['\"]?/",
    r"\bopen\s*\(\s*['\"]/[^'\"]*['\"]\s*,\s*['\"]w",
]
_INTERPRETERS = frozenset({"python", "python3", "perl", "ruby", "node", "sh", "bash", "zsh"})

# The subset that takes a SHELL command string via -c. Only these payloads are
# re-judged by _deny_reason: in Python/Perl/Ruby, ';' separates statements and
# re-applying shell rules would reject ordinary analysis code.
_SHELL_INTERPRETERS = frozenset({"sh", "bash", "zsh"})

# Commands that run some OTHER command as their next token, so the real payload
# is that token. "env rm -rf /" must be judged exactly as "rm -rf /" is.
# xargs is deliberately NOT here: it APPENDS stdin arguments to the command
# rather than passing them through, so "unwrap one token" is the wrong model.
# There is no legitimate use for it in this agent, so it is denied outright.
_WRAPPER_CMDS = frozenset({
    "env", "sudo", "command", "nohup", "time", "stdbuf", "nice", "ionice", "exec",
})

# `make` is not itself destructive, but these targets wipe build trees. Blocking
# all of `make` would break ordinary builds, so only the deleting targets go.
_DESTRUCTIVE_MAKE_TARGETS = frozenset({"clean", "distclean", "realclean"})

# Turns a file list into arguments for an arbitrary command. No legitimate use
# in this agent, and it cannot be modelled by unwrapping a single token.
_DENIED_CMDS = frozenset({"xargs"})

# Ruby / Node / Perl destruction spellings, for the same reason as _INLINE_DESTRUCTION.
_SCRIPTED_DESTRUCTION = [
    r"\brm_rf\b", r"\brm_r\b", r"\brmSync\b", r"\brmdirSync\b",
    r"\bunlink\b", r"\bFileUtils\b", r"\bDir\s*\.\s*rm\b",
]

# Command-line patterns that have no legitimate use here and are always denied.
# Kept as regexes because these are about command *shape*, not target path.
_FORBIDDEN_PATTERNS = [
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bfind\b[^|;&]*\s-delete\b",       # find / -delete
    r"\bfind\b[^|;&]*\s-exec\s+(rm|shred|unlink)",  # find / -exec rm {} +
    r"\bgit\s+clean\b[^|;&]*-[a-z]*f",  # git clean -fdx
    r"\bcurl\b[^|;&]*\|\s*(ba)?sh\b",
    r"\bwget\b[^|;&]*\|\s*(ba)?sh\b",
    r"\bbase64\s+-d\b[^|;&]*\|\s*(ba)?sh\b",
    r"\beval\s*\(",
]


class ToolError(Exception):
    """Raised when a tool execution fails."""


def _safe_resolve(path_str: str) -> Path:
    """Resolve a path, keeping it inside the working directory tree."""
    p = Path(path_str).expanduser()
    # If relative, resolve against cwd
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    # No traversal above cwd is enforced by resolve()
    return p


def _is_inside_workspace(target: Path) -> bool:
    """True if `target` is strictly below the working directory.

    Strictly: the workspace root itself is NOT inside itself, so a destructive
    command aimed at the root (rm -rf .) is denied along with one aimed outside.
    """
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return False
    return cwd in target.parents


def _deny_reason(command: str) -> str | None:
    """Return a human-readable reason if `command` is denied, else None.

    Layered, cheapest first:
      1. shape patterns  — command shape (piped install, find -delete, ...)
      2. chaining        — no `&&`, `;`, `|`, backtick or `$(`
      3. target boundary — destructive commands may only touch the workspace

    Why boundaries rather than a command denylist: with shell=True the shell
    expands `~`, `$HOME` and globs BEFORE any per-command check can inspect
    the result, so no list of forbidden commands can be made complete. A
    destructive command aimed outside the working directory is denied whatever
    its name, and naming interpreters lets inline code be checked too.
    """
    # 1. shape
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return "Forbidden pattern detected. Re-think the command."

    try:
        parts = shlex.split(command)
    except ValueError:
        return "Could not parse command. Re-think the syntax."

    # 2. chaining / substitution — only OUTSIDE quotes, so legitimate inline
    #    code such as python -c 'a and b' is not caught.
    bare = re.sub(r"'[^']*'|\"[^\"]*\"", " ", command)
    for ch in _SHELL_CHAIN_CHARS:
        if ch in bare:
            if ch == "$":
                return (
                    "Variable expansion is not allowed. A path like $HOME is "
                    "resolved by the shell, so it cannot be verified here — "
                    "write the literal path instead."
                )
            return (
                f"Shell metacharacter {ch!r} is not allowed. "
                "Run a single command; do not chain or pipe."
            )

    if not parts:
        return None

    head = Path(parts[0]).name

    # 0. unwrap wrapper commands: "env rm -rf /" is judged as "rm -rf /".
    #    Each wrapper takes options of its own before the real command, so skip
    #    any leading flags — otherwise "env -i rm -rf /" would judge "-i" as the
    #    command and let "rm -rf /" through unexamined.
    #    Bounded to a few levels so `env env env ...` cannot spin.
    for _ in range(4):
        if head not in _WRAPPER_CMDS or len(parts) < 2:
            break
        rest = parts[1:]
        # `sudo -u root cmd` — the flag's value is a separate token, so skip both.
        idx = 0
        while idx < len(rest) and rest[idx].startswith("-"):
            idx += 1
            if head == "sudo" and rest[idx - 1] in ("-u", "-g", "-U", "-C", "-p", "-r", "-t"):
                idx += 1  # this flag takes a value
        if idx >= len(rest):
            break
        parts = rest[idx:]
        head = Path(parts[0]).name

    # 0c. commands with no legitimate use here, whatever their arguments.
    if head in _DENIED_CMDS:
        return f"{head!r} is not available in this environment."

    # 0b. `sh -c '<command>'` runs its quoted argument as a shell command, so
    #     judge that payload with this same policy rather than regex-scanning it.
    if head in _SHELL_INTERPRETERS and len(parts) >= 3 and parts[1] in ("-c",):
        payload = parts[2]
        if payload and payload != command:
            nested = _deny_reason(payload)
            if nested is not None:
                return f"Inner command denied — {nested}"

    # 3a. inline interpreter code
    if head in _INTERPRETERS:
        for pat in _INLINE_DESTRUCTION + _SCRIPTED_DESTRUCTION:
            if re.search(pat, command):
                return (
                    f"Inline code from {head!r} contains a destructive call. "
                    "Run a single, explicit file operation instead."
                )

    # 3b. destructive command targets
    if head in _DESTRUCTIVE_CMDS:
        for arg in parts[1:]:
            if arg.startswith("-"):
                continue
            # "/" and "/*" are the root — always denied, never inside a workspace.
            if arg.rstrip("*").rstrip("/") == "":
                return (
                    f"Refusing to run {head!r} against the filesystem root."
                )
            try:
                resolved = Path(arg).expanduser().resolve()
            except (OSError, RuntimeError):
                continue
            if not _is_inside_workspace(resolved):
                return (
                    f"Refusing to run {head!r} against {resolved}: "
                    "outside the working directory."
                )
            for prot in _SYSTEM_TARGETS:
                if resolved == Path(prot) or Path(prot) in resolved.parents:
                    return (
                        f"Refusing to run {head!r} against protected "
                        f"target {resolved}."
                    )

    # 3c. build systems that delete on request. `make` itself is fine, so only
    # the deleting targets are denied — blocking all of `make` would break
    # ordinary builds, which this agent legitimately runs.
    if head == "make":
        for arg in parts[1:]:
            if arg.startswith("-"):
                continue
            target = arg.split("=", 1)[0]
            if target in _DESTRUCTIVE_MAKE_TARGETS:
                return (
                    f"'make {target}' deletes build output. "
                    "Remove the specific paths instead."
                )
    return None


def _bash(command: str) -> str:
    """Run a single shell command safely. Returns stdout or error text."""
    reason = _deny_reason(command)
    if reason is not None:
        return f"[ToolError] {reason}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT,
            cwd=str(Path.cwd()),
        )
        out = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            out = f"[exit {result.returncode}] " + out
        return out[:_MAX_OUTPUT]
    except subprocess.TimeoutExpired:
        return f"[ToolError] Command timed out after {_BASH_TIMEOUT}s"
    except Exception as exc:
        return f"[ToolError] {type(exc).__name__}: {exc}"


def _read(file_path: str) -> str:
    try:
        p = _safe_resolve(file_path)
        return p.read_text("utf-8", errors="replace")[:_MAX_OUTPUT]
    except FileNotFoundError:
        return f"[ToolError] File not found: {file_path}"
    except IsADirectoryError:
        return f"[ToolError] {file_path} is a directory. Use Glob instead."
    except Exception as exc:
        return f"[ToolError] {type(exc).__name__}: {exc}"


def _write(file_path: str, content: str) -> str:
    try:
        p = _safe_resolve(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = None, ""
        import tempfile as _tmp
        fd, tmp = _tmp.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, p)
        except Exception:
            if tmp:
                os.unlink(tmp)
            raise
        return f"[ok] Wrote {len(content)} chars to {p}"
    except Exception as exc:
        return f"[ToolError] {type(exc).__name__}: {exc}"


def _glob(pattern: str) -> str:
    try:
        matches = list(Path.cwd().rglob(pattern))
        if not matches:
            matches = list(Path.cwd().glob(pattern))
        lines = [str(m.relative_to(Path.cwd())) for m in matches[:50]]
        if not lines:
            return "[no matches]"
        return "\n".join(lines)
    except Exception as exc:
        return f"[ToolError] {type(exc).__name__}: {exc}"


def _grep(pattern: str, path: str) -> str:
    try:
        p = _safe_resolve(path)
        if p.is_file():
            texts = [(p, p.read_text("utf-8", errors="replace"))]
        else:
            texts = []
            for f in p.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        texts.append((f, f.read_text("utf-8", errors="replace")))
                    except Exception:
                        pass

        results = []
        compiled = re.compile(pattern, re.IGNORECASE)
        for f, text in texts:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if compiled.search(line):
                    rel = str(f.relative_to(Path.cwd()) if f.is_relative_to(Path.cwd()) else f)
                    results.append(f"{rel}:{i+1}: {line.rstrip()}")
                    if len(results) >= 20:
                        break
            if len(results) >= 20:
                break

        if not results:
            return "[no matches]"
        return "\n".join(results[:_MAX_OUTPUT // 100])
    except Exception as exc:
        return f"[ToolError] {type(exc).__name__}: {exc}"


_TOOL_MAP = {
    "Bash": _bash,
    "Read": _read,
    "Write": _write,
    "Glob": _glob,
    "Grep": _grep,
}


# ─── Core agent loop ───────────────────────────────────────────

async def _run_agent_loop(
    prompt: str,
    options: ClaudeAgentOptions,
) -> AsyncIterator[AssistantMessage | ResultMessage]:
    """Multi-turn agent loop with streaming + tool execution."""
    from openai import OpenAI, APIError, APIConnectionError

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        or "ollama"
    )
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:11434")
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    model = options.model or "deepseek-v4-flash:cloud"
    max_turns = options.max_turns
    system_prompt = options.system_prompt or "You are a helpful assistant."

    # Subagent descriptions — injected into system prompt
    agent_section = ""
    if options.agents:
        agent_section = "\n\n## Available sub-agents\nYou can delegate to these specialists:\n"
        for name, agent_def in options.agents.items():
            agent_section += f"- **{name}**: {agent_def.description}\n"

    tools_schemas = _build_tools(options.allowed_tools)
    session_id = uuid.uuid4().hex[:16]
    start = time.time()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt + agent_section},
        {"role": "user", "content": prompt},
    ]

    client = OpenAI(api_key=api_key, base_url=base_url)

    for turn_num in range(1, max_turns + 1):
        full_text = ""
        tool_call_deltas: dict[int, dict] = {}

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schemas,
                stream=True,
                temperature=0.7,
            )
        except (APIError, APIConnectionError) as exc:
            yield ResultMessage(
                num_turns=turn_num,
                is_error=True,
                errors=f"APIError: {exc}",
                session_id=session_id,
                duration_ms=int((time.time() - start) * 1000),
            )
            return
        except Exception as exc:
            yield ResultMessage(
                num_turns=turn_num,
                is_error=True,
                errors=f"{type(exc).__name__}: {exc}",
                session_id=session_id,
                duration_ms=int((time.time() - start) * 1000),
            )
            return

        # ── Streaming phase ──────────────────────────────────
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Text streaming
            if delta.content:
                text = delta.content
                full_text += text
                yield AssistantMessage(
                    content=[TextBlock(text=text)],
                    model=model,
                )

            # Tool call deltas (accumulate by index)
            for tc_delta in delta.tool_calls or []:
                idx = tc_delta.index
                slot = tool_call_deltas.setdefault(idx, {
                    "id": "",
                    "function": {"name": "", "arguments": ""},
                })
                if tc_delta.id:
                    slot["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        slot["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        slot["function"]["arguments"] += tc_delta.function.arguments

        # ── Decide: done or execute tools ──────────────────
        if not tool_call_deltas:
            # No tool calls requested — agent is done
            yield ResultMessage(
                num_turns=turn_num,
                is_error=False,
                result=full_text,
                session_id=session_id,
                duration_ms=int((time.time() - start) * 1000),
            )
            return

        # ── Execute tools ────────────────────────────────────
        tool_results: list[dict] = []
        for i in sorted(tool_call_deltas):
            tc = tool_call_deltas[i]
            name = tc["function"]["name"]
            call_id = tc["id"]
            arguments = tc["function"]["arguments"]

            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                result_text = f"[ToolError] Invalid JSON in arguments: {arguments[:200]}"
                tool_results.append({
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": name,
                    "content": result_text,
                })
                continue

            handler = _TOOL_MAP.get(name)
            is_error = False
            if handler is None:
                result_text = f"[ToolError] Unknown tool: {name}"
                is_error = True
            else:
                yield ToolUseMessage(
                    name=name,
                    arguments=args,
                    tool_use_id=call_id,
                    turn=turn_num,
                )
                # Run tool in thread pool (all handlers are sync)
                try:
                    result_text = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: handler(**args)
                    )
                except Exception as exc:
                    result_text = f"[ToolError] {type(exc).__name__}: {exc}"
                    is_error = True

            yield ToolResultMessage(
                name=name,
                result=result_text,
                tool_use_id=call_id,
                turn=turn_num,
                is_error=is_error,
            )

            tool_results.append({
                "tool_call_id": call_id,
                "role": "tool",
                "name": name,
                "content": result_text,
            })

        # Build assistant message with tool_calls for history
        assistant_msg: dict = {"role": "assistant", "content": full_text}
        if tool_call_deltas:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_call_deltas.values()
            ]
        messages.append(assistant_msg)

        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["content"],
            })

    # ── Max turns exceeded ───────────────────────────────
    yield ResultMessage(
        num_turns=max_turns,
        is_error=True,
        errors=f"Agent reached max turns ({max_turns}) without completing.",
        session_id=session_id,
        duration_ms=int((time.time() - start) * 1000),
    )


# ─── Public entry point (unchanged contract) ─────────────────────

def query(
    prompt: str,
    options: ClaudeAgentOptions,
) -> AsyncIterator[AssistantMessage | ResultMessage]:
    """Run the agent loop and stream chunks.

    Yields:
        AssistantMessage — text content as it streams in
        ResultMessage    — final metadata when the loop completes
    """
    return _run_agent_loop(prompt, options)
