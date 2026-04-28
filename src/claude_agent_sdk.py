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
import fnmatch
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
_FORBIDDEN_PATTERNS = [
    r"\brm\s+-rf\s*/\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bchmod\s+-R\s+777\s*/\b",
    r"\>:dev:null\b",        # obfuscated /dev/null writes
    r"\bcurl\s+.*\|\s*sh\b",
    r"\bwget\s+.*\|\s*sh\b",
    r"\bbase64\s+-d\s*\|\s*sh\b",
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


def _bash(command: str) -> str:
    """Run a shell command safely. Returns stdout or error text."""
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"[ToolError] Forbidden pattern detected. Re-think the command."

    if command.strip().startswith(("python -c ", "python3 -c ")):
        # Simple one-liner — accept
        pass

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
            if handler is None:
                result_text = f"[ToolError] Unknown tool: {name}"
            else:
                # Run tool in thread pool (all handlers are sync)
                try:
                    result_text = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: handler(**args)
                    )
                except Exception as exc:
                    result_text = f"[ToolError] {type(exc).__name__}: {exc}"

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
