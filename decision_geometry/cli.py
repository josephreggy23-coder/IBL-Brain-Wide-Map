"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze decision geometry in a public IBL Neuropixels session."
    )
    parser.add_argument("--cache", default="data/cache/session_population.npz")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--force-stream", action="store_true")
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--max-units", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7, help="Random seed for cross-validation splits.")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = run_pipeline(
        cache_path=arguments.cache,
        output_dir=arguments.output_dir,
        force_stream=arguments.force_stream,
        bin_size=arguments.bin_size,
        max_units=arguments.max_units,
        seed=arguments.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
