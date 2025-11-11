"""Command-line entry point for the benchmark toolkit."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_tools import BenchmarkRunner, load_benchmark_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suites defined in YAML configs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML or JSON configuration file describing the benchmark run.",
    )
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    runner = BenchmarkRunner(config)
    summary = runner.run()

    frame = summary.dataframe()
    print(frame.to_string(index=False))  # noqa: T201
    if config.output:
        output_dir = Path(config.output.directory).resolve()
        print(f"Results saved under: {output_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
