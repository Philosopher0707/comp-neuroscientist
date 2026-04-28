#!/usr/bin/env bash
# Run the Comp-Neuroscientist TUI (Go)
set -euo pipefail

cd "$(dirname "$0")/.." || { echo "Failed to cd to project root"; exit 1; }

# Build the Go TUI binary
echo "Building TUI..."
(cd tui && go build -o ../bin/comp-neuro-tui .) || { echo "Build failed"; exit 1; }

echo ""
echo "Starting Comp-Neuroscientist TUI..."
echo "Make sure Ollama is running (http://localhost:11434)"
echo ""
exec ./bin/comp-neuro-tui
