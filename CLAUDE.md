# CLAUDE.md — Comp-Neuroscientist Project

## Project Overview

Autonomous computational neuroscience agent. Describe an analysis task in natural language; the agent loads data, runs analysis pipelines (fMRI, EEG, ephys, calcium, encoding, simulation), and produces a report.

**Stack:**
- Python 3.11+ (agent core, ML/neuro stack)
- Go 1.26+ (TUI via Charm Bubble Tea)
- Ollama (model inference)

**Default model:** `deepseek-v4-flash:cloud` (configurable via `CN_MODEL`)
**Local default (--local):** `llama3.1` (set `CN_LOCAL=true` for privacy)

## Key Files

| File | Purpose |
|---|---|
| `src/claude_agent_sdk.py` | Custom multi-turn agent loop — `query()` returns async generator of chunks |
| `src/comp_neuroscientist/agent.py` | `run_agent()` — env setup, output dirs, JSON mode for Go TUI |
| `src/comp_neuroscientist/cli.py` | CLI entry point with `--json` flag for subprocess IPC |
| `src/comp_neuroscientist/config.py` | Pydantic config with `CN_*` env var overrides |
| `src/comp_neuroscientist/subagents.py` | 8 neuro-specific subagent definitions |
| `tui/main.go` | Go binary entry point — detects python3, verifies agent module |
| `tui/ui/model.go` | Bubble Tea model — streaming, Enter-to-submit, markdown render, path sanitization |
| `tui/ui/styles.go` | Lip Gloss styles (Tokyo Night palette), mode-aware status bar |
| `tui/ui/renderer.go` | Markdown-to-ANSI renderer (bold, italic, code, headers, lists) |
| `tui/agent/agent.go` | Python subprocess manager |
| `tui/protocol/types.go` | JSON event protocol types |

## Architecture

```
User
  ├── CLI: python -m comp_neuroscientist.cli "prompt"
  │         → agent.py → claude_agent_sdk.py → Ollama
  │
  └── TUI: ./bin/comp-neuro-tui
            → runs Python subprocess with --json
            → reads JSON events from stdout
            → renders with Bubble Tea + Lip Gloss
```

The Python agent emits JSON events to stdout:
```json
{"type":"text","content":"analyzing..."}
{"type":"tool_call","name":"Bash","arguments":{"command":"python ..."}}
{"type":"result","text":"...","turns":3,"duration_ms":12500,"is_error":false}
```

The Go TUI reads these and updates the viewport in real time.

## Common Tasks

### Run TUI
```bash
cd /Users/philosopher/active/comp-neuroscientist
cd tui && go build -o ../bin/comp-neuro-tui . && ../bin/comp-neuro-tui
```

### Run CLI
```bash
PYTHONPATH=src \
  python -m comp_neuroscientist.cli "Load BOLD data and compute connectivity"
```

### Run tests
```bash
# Python tests
PYTHONPATH=src python -m pytest tests/ -v

# Go tests
cd tui && go test ./... -v
```

## Critical Rules

- **Never use litellm** — Ollama natively supports OpenAI-compatible API
- **Never set `ANTHROPIC_BASE_URL`** — the CLI and TUI never route on it. The endpoint comes from `Config.ollama_url`, which reads `OLLAMA_BASE_URL` (default `http://localhost:11434`). `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` are consulted only as a credential fallback in non-local mode; they never route traffic. (The SDK still has a legacy `env_fallback=True` branch that reads it, but that branch is unreachable from the CLI/TUI: both always pass `base_url` explicitly.)
- **Never modify tool schemas** without updating `_build_tools()` in `claude_agent_sdk.py`
- **JSON protocol is append-only** — each event is exactly one line of JSON
- **Config is the single endpoint authority** — every CLI/TUI run passes `ClaudeAgentOptions.base_url` explicitly, so the SDK's `if options.base_url:` branch always wins and ambient env cannot route traffic. This is what makes `CN_LOCAL` un-defeatable by a pre-existing `ANTHROPIC_BASE_URL`.

## Dependencies

```
Python (core): openai>=1.0, pydantic>=2.0, numpy>=1.26, scipy>=1.13,
               compneuro-toolkit>=2.0
Python (opt):  tiktoken>=0.8

Go: github.com/charmbracelet/bubbletea,
    github.com/charmbracelet/bubbles,
    github.com/charmbracelet/lipgloss
```

## Subagents

| Name | Domain | Key libs |
|---|---|---|
| fmri | fMRI/BOLD | compneuro.fmri, nibabel, nilearn |
| eeg | EEG/MEG | compneuro.eeg, mne-python |
| ephys | Electrophysiology | compneuro.ephys, spikeinterface |
| calcium | Calcium imaging | compneuro.calcium, suite2p |
| encoding | Encoding models | compneuro.encoding, sklearn |
| simulation | Spiking networks | compneuro.simulation |
| stats | Neuro statistics | compneuro.stats, statsmodels |
| ml | ML/decoding | compneuro.ml, sklearn |

## Known Patterns

- **Reset accumulators on retry**: `collected_text = ""` at top of each attempt
- **Append-only for persistence**: JSONL is append-only, never read-modify-write
- **Flags for flow control**: use `_got_result` flags instead of break-position dependency
