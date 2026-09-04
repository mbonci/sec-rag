"""Financial Document QA — local RAG demo.

Setup (one-time):
    python scripts/standardize_output.py
    python scripts/build_rag_index.py

Run:
    python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import src.constants  # sets HF_HOME before any HuggingFace import

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.generation import load_model, generate
from src.retrieval import search

INDEX_PATH    = Path("outputs/final/index.faiss")
METADATA_PATH = Path("outputs/final/index_metadata.parquet")
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
TOP_K         = 5


def main() -> None:
    for p in (INDEX_PATH, METADATA_PATH):
        if not p.exists():
            sys.exit(f"Missing {p} — run scripts/build_rag_index.py first")

    print("Loading index and metadata ...")
    index = faiss.read_index(str(INDEX_PATH))
    metadata_df = pd.read_parquet(METADATA_PATH)

    print(f"Loading embedding model ({EMBED_MODEL}) ...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    tokenizer, model = load_model()

    print("\nFinancial Document QA  (Ctrl-C to exit)\n")

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue

        chunks = search(query, index, metadata_df, embed_model, k=TOP_K)
        answer = generate(query, chunks, tokenizer, model)

        print(f"\nAnswer: {answer}\n")
        print("Sources:")
        for i, chunk in enumerate(chunks, 1):
            company = chunk.get("company_name", "Unknown")
            year    = chunk.get("filing_year", "?")
            heading = chunk.get("heading", "")
            print(f"  [{i}] {company} ({year}) — {heading}")
        print()


if __name__ == "__main__":
    main()
