"""Export a reproducible evaluation sample for manual review."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants
from src.analyze_results import build_eval_sample, save_eval_sample
from src.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an evaluation sample for manual review.")
    parser.add_argument(
        "--input",
        type=Path,
        default=constants.OUTPUT_JSONL,
        help="JSONL file produced by run_pipeline.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=constants.EVALUATION_DIR / "eval_sample.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=constants.EVAL_SAMPLE_SIZE,
        help="Number of chunks to sample.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"File not found: {args.input}")
        sys.exit(1)

    chunks = read_jsonl(args.input)
    df = build_eval_sample(chunks, n=args.n)
    save_eval_sample(df, path=args.output)
    print(df[["section_category", "section_heading", "chunk_text_preview"]].to_string(index=False))


if __name__ == "__main__":
    main()
