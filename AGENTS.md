# Comp-Neuroscientist — Architecture & Design

## Architecture Overview

```
User (CLI or Go TUI)
    │
    ├── CLI Mode: python -m comp_neuroscientist.cli "prompt"
    │       │
    │       ▼
    │   ┌─────────────────────────────────────────────┐
    │   │       agent.py: run_agent()                 │
    │   │  • Sets up env, output dirs                 │
    │   │  • Calls query() from claude_agent_sdk.py   │
    │   │  • consume_stream() — prints to terminal    │
    │   └──────────────────┬──────────────────────────┘
    │
    └── TUI Mode: Go binary (Charm Bubble Tea)
            │
            ▼
    ┌──────────────────────────────────────────────┐
    │   tui/main.go: Bubble Tea program             │
    │  • Spawns python3 -m comp_neuroscientist.cli  │
    │    --json "prompt" as subprocess              │
    │  • Reads JSON events from stdout line by line │
    │  • Renders streaming text via Lip Gloss       │
    └──────────────────┬───────────────────────────┘
                       │ spawns
                       ▼
    ┌──────────────────────────────────────────────┐
    │ claude_agent_sdk.py: _run_agent_loop()        │
    │  • OpenAI SDK → Ollama /v1/chat/completions   │
    │  • Multi-turn: text + tool execution          │
    │  • Yields AssistantMessage + ResultMessage    │
    └──────────────────┬───────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────┐
    │           Ollama (OpenAI-compatible API)       │
    │  • /v1/chat/completions with streaming        │
    │  • Function-calling for tool execution        │
    └──────────────────────────────────────────────┘
```

### Key Design Decision: Hybrid Go/Python Architecture

| Layer | Technology | Why |
|---|---|---|
| **TUI** | Go + Charm (Bubble Tea, Lip Gloss) | Gold standard for modern terminal UIs. Native SIGWINCH handling, compositor-based rendering (flicker-free), built-in mouse support. |
| **Agent** | Python (custom loop) | Access to the full Python ML/neuro ecosystem: compneuro-toolkit, nilearn, mne-python, spikeinterface, nibabel, sklearn. |
| **IPC** | JSON over stdin/stdout | Simple, debuggable, no dependency. The agent writes one JSON object per line. The Go TUI reads and renders. |

### Why not a Python TUI (Textual)?

The data-scientist project uses Textual for its TUI. For comp-neuroscientist, the Go approach was chosen because:
- **Charm Bubble Tea** is the gold standard for terminal UIs — used by tools like Glow, Lazygit, and K9s
- **Lip Gloss** provides CSS-like styling that's more predictable than Textual CSS
- **Binary distribution** — the Go binary is self-contained, no Python TUI deps needed
- **Performance** — Go's rendering is faster for real-time streaming

## Key Source Files

| File | Purpose |
|---|---|
| `src/claude_agent_sdk.py` | Custom agent loop — `query()` returns async generator |
| `src/comp_neuroscientist/agent.py` | `run_agent()` — env setup, JSON mode, `consume_stream()` |
| `src/comp_neuroscientist/cli.py` | CLI entry point, `--json` flag for Go TUI |
| `src/comp_neuroscientist/config.py` | Pydantic config with env var overrides (`CN_*`) |
| `src/comp_neuroscientist/subagents.py` | 8 neuro-specific subagent definitions |
| `tui/main.go` | Go binary entry point |
| `tui/ui/model.go` | Bubble Tea model, view, event handling |
| `tui/ui/styles.go` | Lip Gloss styles (Tokyo Night palette) |
| `tui/agent/agent.go` | Python subprocess manager |
| `tui/protocol/types.go` | JSON event protocol types |

## Agent Loop Lifecycle

```
_run_agent_loop(prompt, options):
  1. Create OpenAI client → Ollama base_url
  2. Build messages (system + user)
  3. For each turn (1..max_turns):
     a. POST /v1/chat/completions?stream=true
     b. Stream chunks, yield AssistantMessage per delta
     c. Accumulate tool call deltas by index
     d. If no tool calls → yield ResultMessage(done), return
     e. If tool calls → execute each, append results, loop
  4. Max turns → yield ResultMessage(error)
```

## JSON Protocol (Go TUI ↔ Python Agent)

The Python agent in `--json` mode writes one JSON line per event:

```json
{"type":"text","content":"analyzing fMRI data..."}
{"type":"tool_call","name":"Bash","arguments":{"command":"python ..."}}
{"type":"tool_result","name":"Bash","result":"[exit 0] done"}
{"type":"result","text":"...","turns":3,"duration_ms":12500,"is_error":false}
{"type":"error","message":"API connection failed"}
```

The Go TUI reads these lines from the subprocess stdout and updates the UI.

## Subagent System

| Subagent | Domain | Key Modules |
|---|---|---|
| **fmri** | fMRI/BOLD | compneuro.fmri, nibabel, nilearn |
| **eeg** | EEG/MEG | compneuro.eeg, mne-python |
| **ephys** | Electrophysiology / spikes | compneuro.ephys, spikeinterface |
| **calcium** | Calcium imaging | compneuro.calcium, suite2p |
| **encoding** | Encoding models | compneuro.encoding, sklearn |
| **simulation** | Spiking networks | compneuro.simulation, brian2 |
| **stats** | Neuro statistics | compneuro.stats, statsmodels |
| **ml** | ML / decoding | compneuro.ml, sklearn |

## Configuration

| Field | Env var | Default | Description |
|---|---|---|---|
| `model` | `CN_MODEL` | `deepseek-v4-flash:cloud` (or `llama3.1` if `CN_LOCAL`) | Ollama model |
| `ollama_url` | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server — **the only** endpoint authority |
| `local` | `CN_LOCAL` | unset (cloud) | `1`/`true`/`yes` forces local-only: loopback endpoint + `llama3.1` default |
| `max_turns` | `CN_MAX_TURNS` | `30` | Max agent iterations |
| `effort` | `CN_EFFORT` | `high` | Agent effort level |
| `output_dir` | `CN_OUTPUT_DIR` | `results` | Output directory |
| `api_key` (derived) | `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | `ollama` in local mode | Credential fallback for non-local mode only; never used for endpoint routing |

`ANTHROPIC_BASE_URL` never routes traffic on the CLI/TUI path — do not set it. The
endpoint comes from `Config.ollama_url` (env `OLLAMA_BASE_URL`, default
`http://localhost:11434`), normalized by `config.resolved_base_url` and passed to
the SDK as `ClaudeAgentOptions.base_url`. Because both the CLI and the TUI always
pass that value explicitly, the SDK's `if options.base_url:` branch always wins and
ambient env cannot reroute the call. (The SDK still keeps a legacy
`env_fallback=True` branch that reads `ANTHROPIC_BASE_URL` for third-party callers
who have not threaded a value through; that branch is unreachable from this repo's
own entry points.) That single-authority chain is what makes `CN_LOCAL`
un-defeatable by ambient environment variables.

## TUI Details

The Go TUI uses [Bubble Tea](https://github.com/charmbracelet/bubbletea) (Elm architecture) + [Lip Gloss](https://github.com/charmbracelet/lipgloss) (styling):

- **Layout:** Sidebar (30% width) + main content + input bar + status bar
- **Streaming:** Real-time text display via viewport with auto-scroll
- **Input:** textinput with Enter-to-submit, Esc to focus output
- **Signals:** SIGWINCH handled natively by Bubble Tea
- **Mouse:** `tea.WithMouseCellMotion()` for click-to-focus, scroll
- **Flicker-free:** Bubble Tea compositor sends only changed characters

### Keyboard shortcuts

| Key | Action |
|---|---|
| Enter | Submit prompt |
| Esc | Focus output pane |
| ^E | Focus input |
| ^L | Clear output |
| ^C / q | Quit (stops agent first if running) |
| ^Z | Stop agent |
| ? | Toggle help |
| ↑↓ | Scroll output / history |

## Building & Running

```bash
# Build Go TUI
cd tui && go build -o ../bin/comp-neuro-tui .

# Run TUI
./bin/comp-neuro-tui

# Run CLI
PYTHONPATH=src python -m comp_neuroscientist.cli "Run fMRI analysis"
```
