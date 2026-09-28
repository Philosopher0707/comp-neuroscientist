"""
Agent entry point — sets up environment, runs the agent loop, consumes the stream.
Supports two output modes:
  • Terminal (default): prints streaming text, writes report.md
  • JSON mode (--json): writes JSON events to stdout for the Go TUI
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator, List, Optional

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseMessage,
    ToolResultMessage,
)

from .config import config
from .subagents import ALL_SUBAGENTS


# ─── System prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are Comp-Neuroscientist, an autonomous computational neuroscience agent.
You specialize in analyzing fMRI, EEG, MEG, calcium imaging, electrophysiology data,
building encoding models, running spiking network simulations, and performing
statistical analyses on neuroimaging data.

## Your capabilities
- Load and preprocess neuroimaging data (compneuro-toolkit, nibabel, mne-python, etc.)
- Run full analysis pipelines: preprocessing → statistics → visualization → interpretation
- Train encoding/decoding models with proper cross-validation
- Generate publication-quality figures (saved to results/plots/)
- Write markdown reports (saved to results/report.md)

## Rules
- Always work from the current working directory
- Save output files to the `results/` directory (plots, models, processed data)
- Use sub-agents for specialized tasks when appropriate
- Explain your reasoning as you go
- If you hit an error, try an alternative approach before giving up
- Keep tool output focused — don't dump large files to the conversation
- Use Glob to list files before operating on them
"""


# ─── Stream consumer ────────────────────────────────────────────

def consume_stream(
    stream: AsyncIterator[AssistantMessage | ResultMessage],
    json_mode: bool = False,
) -> str:
    """Consume the agent stream and return the final result text.

    In terminal mode: prints text to stdout.
    In JSON mode: writes JSON events to stdout for the Go TUI to consume.
    """
    collected: List[str] = []
    result_status = "running"
    turns = 0

    try:
        while True:
            try:
                msg = asyncio.run(stream.__anext__())
            except StopAsyncIteration:
                break

            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text:
                        if json_mode:
                            _emit_json({"type": "text", "content": block.text})
                        else:
                            print(block.text, end="", flush=True)
                        collected.append(block.text)

                if msg.model:
                    pass  # model info available

            elif isinstance(msg, ToolUseMessage):
                if json_mode:
                    _emit_json({"type": "status", "status": "running", "turns": msg.turn})
                    _emit_json({
                        "type": "tool_call",
                        "name": msg.name,
                        "arguments": msg.arguments,
                    })

            elif isinstance(msg, ToolResultMessage):
                if json_mode:
                    _emit_json({
                        "type": "tool_result",
                        "name": msg.name,
                        "result": msg.result,
                    })

            elif isinstance(msg, ResultMessage):
                turns = msg.num_turns
                if msg.is_error:
                    result_status = "error"
                    if json_mode:
                        _emit_json({"type": "error", "message": msg.errors})
                    else:
                        print(f"\n\n[Error: {msg.errors}]", flush=True)
                else:
                    result_status = "success"

                if json_mode:
                    _emit_json({
                        "type": "result",
                        "text": msg.result,
                        "turns": msg.num_turns,
                        "duration_ms": msg.duration_ms,
                        "is_error": msg.is_error,
                    })
                else:
                    print(f"\n\n[Finished in {msg.duration_ms / 1000:.1f}s, {msg.num_turns} turns]",
                          flush=True)

    except KeyboardInterrupt:
        result_status = "interrupted"
        if json_mode:
            _emit_json({"type": "error", "message": "interrupted"})
        print("\n[Interrupted]", flush=True)

    except Exception as exc:
        result_status = "error"
        if json_mode:
            _emit_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        print(f"\n[Unexpected error: {type(exc).__name__}: {exc}]", flush=True)

    return "".join(collected)


def _emit_json(obj: dict) -> None:
    """Write a JSON event line to stdout, flushed for the Go TUI reader."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ─── Setup environment ─────────────────────────────────────────

def _setup_environment(prompt: str) -> ClaudeAgentOptions:
    """Configure the agent options from config and the prompt's domain."""
    # Determine which agents might be relevant based on keywords in the prompt
    relevant_agents: dict = {}
    prompt_lower = prompt.lower()

    # Always include all agents — the orchestrator decides when to use them
    relevant_agents = dict(ALL_SUBAGENTS)

    # Collect tool names from all subagents
    all_tools: set[str] = set()
    for agent_def in ALL_SUBAGENTS.values():
        all_tools.update(agent_def.tools)
    # Base tools
    all_tools.update(["Bash", "Read", "Write", "Glob", "Grep"])

    return ClaudeAgentOptions(
        max_turns=config.max_turns,
        model=config.model,
        allowed_tools=sorted(all_tools),
        agents=relevant_agents,
        system_prompt=SYSTEM_PROMPT,
    )


# ─── Entry point ────────────────────────────────────────────────

def run_agent(prompt: str, json_mode: bool = False) -> str:
    """Run the agent with the given prompt.

    Args:
        prompt: The user's natural language prompt.
        json_mode: If True, emit JSON events to stdout instead of plain text.

    Returns:
        The full collected text from the agent.
    """
    # Create output directories
    for d in [config.output_dir, config.plots_dir, config.models_dir, config.processed_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    options = _setup_environment(prompt)

    # Set env vars for the agent SDK
    os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "ollama")
    os.environ.setdefault("ANTHROPIC_BASE_URL", config.ollama_url)

    stream = query(prompt, options)
    result_text = consume_stream(stream, json_mode=json_mode)

    # Save report
    if result_text:
        report_path = Path(config.output_dir) / "report.md"
        report_path.write_text(result_text, encoding="utf-8")
        if not json_mode:
            print(f"\nReport saved: {report_path}", flush=True)

    return result_text
