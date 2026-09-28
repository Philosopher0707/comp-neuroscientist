"""
CLI entry point for Comp-Neuroscientist.

Usage:
  # Terminal mode (default: cloud model)
  python -m comp_neuroscientist.cli "Run fMRI analysis on dataset"

  # Local inference (privacy-sensitive data)
  python -m comp_neuroscientist.cli --local "Run fMRI analysis"

  # JSON mode (for Go TUI subprocess)
  python -m comp_neuroscientist.cli --json "Run fMRI analysis"

  # List models
  python -m comp_neuroscientist.cli --list-models
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comp-Neuroscientist — Autonomous Computational Neuroscience Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Load BOLD data and compute functional connectivity"
  %(prog)s --local "Run EEG time-frequency analysis"
  %(prog)s --model glm-5.1:cloud "Spike sorting on neuropixels"
  %(prog)s --local --json "Analysis via TUI"
  %(prog)s --list-models
        """,
    )

    parser.add_argument("prompt", nargs="?", default="",
                        help="Natural language prompt describing the analysis task")

    parser.add_argument("--model", "-m", default=None,
                        help="Ollama model name (overrides CN_MODEL env var)")

    parser.add_argument("--max-turns", type=int, default=None,
                        help="Max agent loop iterations (overrides CN_MAX_TURNS)")

    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (overrides CN_OUTPUT_DIR)")

    parser.add_argument("--json", action="store_true",
                        help="Output JSON events for Go TUI subprocess consumption")

    parser.add_argument("--local", action="store_true",
                        help="Use local model (privacy-first, no cloud API calls)")

    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")

    return parser.parse_args()


def _list_models() -> None:
    """Print available models."""
    print("Local models (privacy-first, no data leaves your machine):")
    print("  llama3.1 (8B)     — Fast general-purpose (default for --local)")
    print("  mistral (7B)      — Strong reasoning for smaller models")
    print("  kimi-k2.5:local   — Long context, good for multi-step")
    print()
    print("Cloud models (require internet, data sent to API):")
    print("  deepseek-v4-flash:cloud  — Fast, general-purpose (default)")
    print("  glm-5.1:cloud           — Strong reasoning for complex analysis")
    print("  kimi-k2.6:cloud         — Latest Kimi model")
    print("  minimax-m2.7:cloud      — Good for structured report generation")


def main() -> None:
    args = _parse_args()

    if args.list_models:
        _list_models()
        sys.exit(0)

    if not args.prompt:
        # Interactive terminal prompt
        try:
            args.prompt = input("Enter your analysis task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

    if not args.prompt:
        print("No prompt provided. Usage: python -m comp_neuroscientist.cli <prompt>")
        sys.exit(1)

    # Apply CLI overrides to config
    from .config import config as cfg

    if args.local:
        cfg.local = True
        if not args.model:
            cfg.model = "llama3.1"

    if args.model:
        cfg.model = args.model
    if args.max_turns is not None:
        cfg.max_turns = args.max_turns
    if args.output:
        cfg.output_dir = args.output

    # Run the agent
    from .agent import run_agent
    run_agent(args.prompt, json_mode=args.json)


if __name__ == "__main__":
    main()
