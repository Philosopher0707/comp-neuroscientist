# Comp-Neuroscientist

**Autonomous Computational Neuroscience Agent** — describe an analysis task in natural language, get a complete fMRI, EEG, electrophysiology, calcium imaging, or simulation analysis with visualizations and a report.

Built on a custom Python agent loop (OpenAI SDK → Ollama) with a Go terminal UI using [Charm](https://charm.sh/) (Bubble Tea + Lip Gloss).

## Quick Start

### 1. Requirements

- Python 3.11+
- Go 1.26+ (for the TUI)
- [Ollama](https://ollama.com/) v0.14.0+ running on port 11434
- The `compneuro-toolkit` library (install from source in `/Users/philosopher/research/compneuro-toolkit`)

### 2. Install

```bash
git clone <repo>
cd comp-neuroscientist

# Python deps
pip install openai pydantic numpy scipy compneuro-toolkit

# Build Go TUI
cd tui && go build -o ../bin/comp-neuro-tui . && cd ..
```

### 3. Run

**CLI mode:**
```bash
PYTHONPATH=src \
ANTHROPIC_AUTH_TOKEN=ollama \
ANTHROPIC_BASE_URL=http://localhost:11434 \
  python -m comp_neuroscientist.cli "Load BOLD data and compute functional connectivity"
```

**TUI mode:**
```bash
./bin/comp-neuro-tui
```

## Features

### Analysis Capabilities

| Domain | What it does |
|---|---|
| **fMRI** | Preprocessing, GLM, ROI extraction, functional connectivity, decoding, RSA |
| **EEG/MEG** | Filtering, ICA, epoching, ERPs, time-frequency, source localization, decoding |
| **Electrophysiology** | Spike sorting (Kilosort, Mountainsort), quality metrics, PSTH, LFP |
| **Calcium imaging** | Suite2p processing, neuropil correction, deconvolution, ensemble detection |
| **Encoding models** | Linear encoding (ridge), TRIBE deep encoding, RSA, noise ceiling |
| **Simulation** | LIF networks, STDP, raster plots, parameter sweeps |
| **Statistics** | Permutation tests with cluster correction, mixed models, FDR |
| **ML decoding** | SVM, logistic regression, nested CV, feature importance, ROC |

### Output Structure

```
results/
├── plots/          # Visualizations (PNG)
├── models/         # Trained model files
├── processed/      # Cleaned/preprocessed data
└── report.md       # Final markdown report
```

### TUI Features

| Feature | Key |
|---|---|
| Submit prompt | Enter |
| Focus input | Ctrl+E |
| Focus output | Esc |
| Clear output | Ctrl+L |
| Stop agent | Ctrl+Z |
| Quit | Ctrl+C / q |
| Help | ? |

The TUI is powered by [Bubble Tea](https://github.com/charmbracelet/bubbletea) (Elm architecture) and [Lip Gloss](https://github.com/charmbracelet/lipgloss) (styling):

- **Fluid layout** — sidebar + main content, adapts to terminal size
- **Real-time streaming** — text appears as the agent generates it
- **Flicker-free** — compositor renders only changed characters
- **Mouse support** — click to focus, scroll output
- **SIGWINCH-aware** — re-renders on terminal resize

## Configuration

| Env var | Flag | Default | Description |
|---|---|---|---|
| `CN_MODEL` | `--model` | `deepseek-v4-flash:cloud` | Ollama model |
| `OLLAMA_BASE_URL` | — | `http://localhost:11434` | Ollama server |
| `CN_MAX_TURNS` | `--max-turns` | `30` | Max agent iterations |
| `CN_EFFORT` | — | `high` | Effort level |
| `CN_OUTPUT_DIR` | `--output` | `results` | Output directory |

## Architecture

```
                    ┌──────────────────────────────┐
                    │     Go TUI (Charm)            │
                    │  main.go → model.go          │
                    │         ↓                    │
User ──► TUI ───────┤  spawns Python subprocess    │
                    │  with --json "prompt"         │
                    │         ↓                    │
                    │  reads JSON events from       │
                    │  stdout, renders viewport     │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Python Agent Core            │
                    │  claude_agent_sdk.py          │
                    │  agent.py                    │
                    │  subagents.py                 │
                    │         ↓                    │
                    │  OpenAI SDK → Ollama API      │
                    │  Multi-turn: text + tools     │
                    └──────────────────────────────┘
```

The Python agent runs as a subprocess of the Go TUI. They communicate via JSON-line protocol over stdout. This hybrid approach combines Go's excellent TUI libraries with Python's rich neuroscience ecosystem.

## Subagents

The agent can delegate specialized work to 8 sub-agents:

| Subagent | Expertise |
|---|---|
| **fmri** | fMRI preprocessing, GLM, connectivity, decoding |
| **eeg** | EEG/MEG filtering, ICA, ERP, time-frequency |
| **ephys** | Spike sorting, quality metrics, PSTH, LFP |
| **calcium** | Calcium trace extraction, deconvolution, ensembles |
| **encoding** | Encoding models, RSA, noise ceiling, model comparison |
| **simulation** | LIF/STDP network simulation, parameter sweeps |
| **stats** | Permutation tests, mixed models, FDR, power analysis |
| **ml** | Decoder pipelines, nested CV, feature importance |

## Running Tests

```bash
# Python tests
PYTHONPATH=src python -m pytest tests/ -v

# Go tests
cd tui && go test ./... -v

# All tests
bash scripts/test.sh
```
