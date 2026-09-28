"""
Agent entry point — sets up environment, runs the agent loop, consumes the stream.
Supports two output modes:
  • Terminal (default): prints streaming text, writes report.md
  • JSON mode (--json): writes JSON events to stdout for the Go TUI
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncIterator, List

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
                if msg.is_error:
                    if json_mode:
                        _emit_json({"type": "error", "message": msg.errors})
                    else:
                        print(f"\n\n[Error: {msg.errors}]", flush=True)

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
        if json_mode:
            _emit_json({"type": "error", "message": "interrupted"})
        print("\n[Interrupted]", flush=True)

    except Exception as exc:
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
    """Configure the agent options from config.

    Args:
        prompt: The user's prompt. Used to select which subagents and tools are
            exposed — see `_select_relevant_agents`.

    Exposing every subagent and the union of every subagent's tools on every run
    was a real but blunt over-grant: a prompt about EEG preprocessing had the
    same tool surface as one that would train encoders and run simulations. Tool
    surface is now scoped to the prompt's actual subject matter.
    """
    relevant_agents = _select_relevant_agents(prompt)

    # Tool surface = tools of the SELECTED agents, not of all agents. Base tools
    # are always present because every task needs to read and write results.
    all_tools: set[str] = set()
    for agent_def in relevant_agents.values():
        all_tools.update(agent_def.tools)
    # Base tools
    all_tools.update(["Bash", "Read", "Write", "Glob", "Grep"])

    return ClaudeAgentOptions(
        max_turns=config.max_turns,
        model=config.model,
        allowed_tools=sorted(all_tools),
        agents=relevant_agents,
        system_prompt=SYSTEM_PROMPT,
        # Config is the single authority for the endpoint. Passing it explicitly
        # means ambient ANTHROPIC_BASE_URL can no longer override --local.
        base_url=config.resolved_base_url,
        api_key=config.api_key,
    )


def _select_relevant_agents(prompt: str) -> dict:
    """Pick the subagents a prompt could plausibly need.

    Selection is keyword-based and deliberately conservative in the safe
    direction: an unrecognised prompt falls back to ALL subagents, so scoping
    never removes a capability the model genuinely needed. The benefit is that a
    clearly-scoped prompt no longer advertises every specialist, which both
    narrows the tool surface and stops the model from delegating to an
    irrelevant specialist.
    """
    if not prompt or not prompt.strip():
        return dict(ALL_SUBAGENTS)

    text = prompt.lower()
    matched = {
        name: agent_def
        for name, agent_def in ALL_SUBAGENTS.items()
        if any(kw in text for kw in _AGENT_KEYWORDS.get(name, ()))
    }
    return matched or dict(ALL_SUBAGENTS)


# ─── Subagent routing ────────────────────────────────────────────

# Keyword → subagent triggers. Used to scope the exposed subagent and tool set
# to the prompt. Keys must match ALL_SUBAGENTS. Matching is substring-based on
# the lowercased prompt, so entries should be unambiguous enough that a
# substring hit means the domain is genuinely in play.
_AGENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fmri": ("fmri", "bold", "mri", "glm", "connectom", "resting-state",
             "resting state", "task-based", "rois", "surface plot",
             "glass brain", "searchlight", "bids", "realign", "smooth"),
    "eeg": ("eeg", "erp", "erp", "meg", "eog", "artifact", "ica",
            "time-frequency", "time frequency", "tfr", "evoked",
            "seizure", "spindle", "montage", "electrode", "channel"),
    "ephys": ("ephys", "electrophysiolog", "patch clamp", "whole-cell",
              "spike train", "voltage clamp", "current clamp", "nanogap",
              "intracellular", "membrane potential", "vhold"),
    "encoding": ("encoding", "decoding", "mvpa", "rsa", "representational",
                 "similarity", "noise ceiling", "voxel", "ridge", "voxelwise",
                 "model comparison", "receptive field"),
    "simulation": ("simulat", "spiking", "lif", "leaky integrate",
                   "stdp", "plasticity", "raster", "population rate",
                   "mean-field", "mean field", "lfp", "snn", "neuron model",
                   "tuning curve"),
    "calcium": ("calcium", "caiman", "suite2p", "two-photon", "2p",
                "gcamp", "neuropil", "oasis", "deconvolution", "calcium trace",
                "cell sorting", "fluorescence"),
    "stats": ("statistic", "statistical", "p-value", "p value", "ttest",
              "t-test", "anova", "mixed-effects", "mixed effects",
              "multiple comparison", "correction", "fdr", "bonferroni",
              "permutation test", "bootstrap", "confidence interval",
              "effect size", "power analysis", "shuffl"),
    "ml": ("machine learning", "classifier", "classification", "svm",
           "logistic regression", "random forest", "gradient boost",
           "xgboost", "sklearn", "scikit", "cross-validation",
           "cross validation", "hyperparameter", "pca", "cca", "umap",
           "autoencoder", "dimensionality reduction", "confusion matrix",
           "roc", "precision", "recall", "f1", "leave-one-subject-out",
           "model training", "fine-tune", "fine tune", "network"),
}

# Guard against drift: a subagent with no keyword entry can never be selected,
# which would be a silent capability loss. Keys are checked against
# ALL_SUBAGENTS at import time so adding an agent without keywords fails loudly.
assert set(_AGENT_KEYWORDS) == set(ALL_SUBAGENTS), (
    "_AGENT_KEYWORDS must cover every subagent exactly once; "
    f"missing={sorted(set(ALL_SUBAGENTS) - set(_AGENT_KEYWORDS))}, "
    f"stale={sorted(set(_AGENT_KEYWORDS) - set(ALL_SUBAGENTS))}"
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

    # The endpoint is carried on `options` (see `_setup_environment`). It is
    # deliberately NOT written into os.environ: mutating the process
    # environment is what made routing invisible and untestable, and any
    # ANTHROPIC_BASE_URL already in the caller's shell would otherwise win.
    # os.environ is untouched from here on.

    stream = query(prompt, options)
    result_text = consume_stream(stream, json_mode=json_mode)

    # Save report
    if result_text:
        report_path = Path(config.output_dir) / "report.md"
        report_path.write_text(result_text, encoding="utf-8")
        if not json_mode:
            print(f"\nReport saved: {report_path}", flush=True)

    return result_text
