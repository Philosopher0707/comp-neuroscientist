#!/usr/bin/env bash
# Run tests for Comp-Neuroscientist
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="src:$PYTHONPATH"

echo "=== Python tests ==="
python3 -m pytest tests/ -v "$@"

echo ""
echo "=== Go tests ==="
cd tui
go test ./... -v
cd ..

echo ""
echo "All tests passed."
