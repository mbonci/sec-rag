"""CLI tool for quick inspection of pipeline outputs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants
from src.inspect_data import load_and_show


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pipeline outputs from a JSONL file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=constants.OUTPUT_JSONL,
        help="JSONL file produced by run_pipeline.py.",
    )
    parser.add_argument(
        "--show",
        choices=["sections", "entities", "tables", "warnings"],
        default="sections",
        help="What to display.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of items to show.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"File not found: {args.input}")
        sys.exit(1)

    load_and_show(args.input, show=args.show, n=args.n)


if __name__ == "__main__":
    main()
