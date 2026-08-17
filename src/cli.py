"""Unified command-line interface: python -m src <command>."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Metaheuristic prompt optimization toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("experiment", help="Run a budgeted optimization experiment.")
    exp.add_argument("--fast", action="store_true")
    exp.add_argument("--balanced", action="store_true")
    exp.add_argument("--research", action="store_true")
    exp.add_argument("--config", type=str, default=None)
    exp.add_argument("--backend", choices=["mock", "ollama"], default=None)
    exp.add_argument("--max_data", type=int, default=None)
    exp.add_argument("--seeds", type=str, default=None)
    exp.add_argument("--datasets", type=str, default=None)
    exp.add_argument("--dspy", action="store_true")
    exp.add_argument("--results-dir", type=str, default=None)

    sub.add_parser("stats", help="Wilcoxon tests on results/runs.jsonl.")
    sub.add_parser("plots", help="Generate publication figures from logged runs.")
    sub.add_parser("inspect", help="Print the best discovered prompts.")
    sub.add_parser("dashboard", help="Launch the Streamlit results dashboard.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, _rest = parser.parse_known_args(argv)

    if args.command == "experiment":
        from src.experiment import main as experiment_main

        forwarded = [a for a in argv[1:] if a != "experiment"]
        experiment_main(forwarded)
        return

    if args.command == "stats":
        from src.stats import run_stats

        run_stats()
        return

    if args.command == "plots":
        from src.plots import generate_all

        generate_all()
        return

    if args.command == "inspect":
        from src.visualize import visualize_best_prompts

        visualize_best_prompts()
        return

    if args.command == "dashboard":
        import subprocess

        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"]))

    parser.print_help()


if __name__ == "__main__":
    main()
