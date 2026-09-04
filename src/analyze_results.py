"""Coverage and quality reporting over the processed chunk dataset."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from src.constants import EVALUATION_DIR, RANDOM_SEED, SUMMARY_JSON
from src.utils import read_jsonl


def build_summary(chunks: list[dict], resolution_map: Optional[list[dict]] = None) -> dict:
    """Compute a quality and coverage summary over chunks."""
    n_docs = len({c["document_id"] for c in chunks})
    chunk_texts = [c["chunk_text"] for c in chunks]
    chunk_lengths = [len(t) for t in chunk_texts]

    # Section category distribution
    cat_counts = Counter(c["section_category"] for c in chunks)

    # Entity stats
    chunks_with_entities = sum(1 for c in chunks if c.get("entities"))
    entity_type_counts: Counter = Counter()
    for c in chunks:
        for e in c.get("entities", []):
            entity_type_counts[e["entity_type"]] += 1

    # Table stats
    chunks_with_tables = sum(1 for c in chunks if c.get("tables"))
    total_tables = sum(len(c.get("tables", [])) for c in chunks)

    # Warning stats
    chunks_with_warnings = sum(1 for c in chunks if c.get("processing_warnings"))

    # Resolution stats
    n_groups = len(resolution_map) if resolution_map else 0

    summary = {
        "documents_processed": n_docs,
        "total_chunks": len(chunks),
        "avg_chunks_per_doc": round(len(chunks) / max(n_docs, 1), 1),
        "avg_chunk_length_chars": round(sum(chunk_lengths) / max(len(chunk_lengths), 1), 0),
        "median_chunk_length_chars": _median(chunk_lengths),
        "section_category_distribution": dict(cat_counts.most_common()),
        "chunks_with_entities_pct": _pct(chunks_with_entities, len(chunks)),
        "entity_counts_by_type": dict(entity_type_counts.most_common()),
        "resolved_entity_groups": n_groups,
        "chunks_with_tables": chunks_with_tables,
        "chunks_with_tables_pct": _pct(chunks_with_tables, len(chunks)),
        "total_tables_found": total_tables,
        "chunks_with_warnings": chunks_with_warnings,
    }
    return summary


def print_summary(summary: dict) -> None:
    """Print the summary to stdout in a readable format."""
    print("=" * 60)
    print("Pipeline Results Summary")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for kk, vv in v.items():
                print(f"    {kk:<40} {vv}")
        else:
            print(f"  {k:<45} {v}")
    print("=" * 60)


def save_summary(summary: dict, path: Path = SUMMARY_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def build_eval_sample(
    chunks: list[dict],
    n: int = 20,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Draw a reproducible stratified sample for manual review."""
    rng = random.Random(seed)
    # Stratify by section_category
    by_cat: dict[str, list[dict]] = {}
    for c in chunks:
        by_cat.setdefault(c["section_category"], []).append(c)

    sample: list[dict] = []
    per_cat = max(1, n // max(len(by_cat), 1))
    for cat_chunks in by_cat.values():
        sample.extend(rng.sample(cat_chunks, min(per_cat, len(cat_chunks))))

    # Top up to n if needed
    remaining = [c for c in chunks if c not in sample]
    rng.shuffle(remaining)
    sample.extend(remaining[: max(0, n - len(sample))])
    sample = sample[:n]

    rows = []
    for c in sample:
        entities_str = "; ".join(
            f"{e['text']} ({e['entity_type']})" for e in c.get("entities", [])[:5]
        )
        resolved_str = "; ".join(
            g["canonical_name"] for g in c.get("resolved_entities", [])[:3]
        )
        table_count = len(c.get("tables", []))
        rows.append({
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "source_filename": c["source_path"].split("/")[-1][:50],
            "section_heading": c["section_heading"][:80],
            "section_category": c["section_category"],
            "chunk_text_preview": c["chunk_text"][:300],
            "entities_preview": entities_str,
            "resolved_entities_preview": resolved_str,
            "table_count": table_count,
            "warnings": "; ".join(c.get("processing_warnings", [])),
            # Human review columns
            "section_correct": "",
            "entities_correct": "",
            "entity_resolution_correct": "",
            "table_parse_correct": "",
            "review_notes": "",
        })

    return pd.DataFrame(rows)


def save_eval_sample(
    df: pd.DataFrame,
    path: Path = EVALUATION_DIR / "eval_sample.csv",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


def flag_suspicious_merges(resolution_map: list[dict], max_show: int = 10) -> pd.DataFrame:
    """Find resolved groups where variants look very different (possible false positives)."""
    from rapidfuzz import fuzz

    suspicious = []
    for g in resolution_map:
        if len(g["variants"]) < 2:
            continue
        variants = g["variants"]
        # Check all pairs
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                score = fuzz.token_sort_ratio(variants[i].lower(), variants[j].lower())
                if score < 70:
                    suspicious.append({
                        "canonical_name": g["canonical_name"],
                        "variant_a": variants[i],
                        "variant_b": variants[j],
                        "similarity": score,
                    })

    df = (
        pd.DataFrame(suspicious)
        .sort_values("similarity")
        .head(max_show)
        .reset_index(drop=True)
        if suspicious
        else pd.DataFrame()
    )
    return df


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2)
