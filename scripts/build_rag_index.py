"""Build a FAISS vector index from the standardized Parquet dataset.

Usage:
    python scripts/build_rag_index.py \
        --input outputs/final/chunks.parquet \
        --output-dir outputs/final/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.constants  # sets HF_HOME before any HuggingFace import

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

METADATA_COLS = [
    "record_id", "document_id", "company_name", "filing_year",
    "heading", "section_category", "chunk_text",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks.parquet.")
    parser.add_argument("--input", type=Path, default=Path("outputs/final/chunks.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/"))
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}  (run standardize_output.py first)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    texts = df["retrieval_text"].tolist()
    print(f"Loaded {len(df)} chunks from {args.input}")

    # ── embeddings (cached) ───────────────────────────────────────────────────
    cache_path = args.output_dir / "embeddings.npy"
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            print(f"Loaded cached embeddings from {cache_path}")
        else:
            print(f"Cache row count mismatch ({embeddings.shape[0]} vs {len(texts)}) — re-embedding")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print(f"Embedding with {EMBED_MODEL} ...")
        model = SentenceTransformer(EMBED_MODEL)
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        np.save(cache_path, embeddings)
        print(f"Embeddings saved → {cache_path}")

    # ── FAISS index ───────────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = args.output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Index saved → {index_path}  ({index.ntotal} vectors, dim={dim})")

    # ── metadata ──────────────────────────────────────────────────────────────
    meta = df[METADATA_COLS].copy()
    meta.insert(0, "faiss_id", range(len(meta)))
    meta_path = args.output_dir / "index_metadata.parquet"
    meta.to_parquet(meta_path, index=False)
    print(f"Metadata saved → {meta_path}")


if __name__ == "__main__":
    main()
