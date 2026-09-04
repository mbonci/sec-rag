"""CLI entry point for the document processing pipeline."""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial document NLP pipeline.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=constants.DATA_DIR,
        help="Directory containing .txt (and optionally .pdf) documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=constants.OUTPUT_JSONL,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of documents to process (default: all).",
    )
    parser.add_argument(
        "--include-pdfs",
        action="store_true",
        default=False,
        help="Also process PDF files (requires PyMuPDF).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help="Run standardization after pipeline and write outputs/final/chunks.parquet.",
    )
    parser.add_argument(
        "--use-llm-table-summaries",
        action="store_true",
        default=True,
        help="Use OpenAI to summarize parsed tables (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=7,
        metavar="N",
        help="Stop after stage N (1-7). E.g. --stop-after 4 skips table parsing and output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    from pathlib import Path as _Path
    standardized = _Path("outputs/final/chunks.parquet") if args.standardize else None

    run_pipeline(
        input_dir=args.input_dir,
        output_jsonl=args.output,
        output_standardized=standardized,
        limit=args.limit,
        include_pdfs=args.include_pdfs,
        use_llm_table_summaries=args.use_llm_table_summaries,
        stop_after=args.stop_after,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
