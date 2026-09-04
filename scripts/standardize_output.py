"""Convert pipeline JSONL output to a clean Parquet dataset for RAG.

Usage:
    python scripts/standardize_output.py \
        --input outputs/intermediate/all_chunks_with_entities_resolved_tables.jsonl \
        --output outputs/final/chunks.parquet
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── helpers ───────────────────────────────────────────────────────────────────

def serialize(value) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False, sort_keys=True)


def parse_source_path(path: str) -> tuple[str, str, str]:
    """Return (company_name, cik, filing_year) parsed from the filename stem."""
    stem = Path(path).name

    company_raw = re.split(r"__CIK", stem)[0]
    company_name = re.sub(r"_+", " ", company_raw).strip()

    cik_m = re.search(r"CIK(\d+)", stem)
    cik = cik_m.group(1).lstrip("0") if cik_m else ""

    # accession numbers encode the filing year: e.g. 000162828018001324 → 2018
    year_m = re.search(r"(\d{4})\d{6}10[xk]", stem, re.IGNORECASE)
    if year_m:
        filing_year = year_m.group(1)
    else:
        yy_m = re.search(r"__\d{10}(\d{2})", stem)
        filing_year = "20" + yy_m.group(1) if yy_m else ""

    return company_name, cik, filing_year


def get_table_summaries(tables: list) -> str:
    return "\n".join(t["summary"] for t in tables if t.get("summary"))


def get_entity_names(entities: list) -> str:
    return ", ".join(e["text"] for e in entities if e.get("text"))


def get_resolved_names(resolved: list) -> str:
    return ", ".join(
        e["canonical_name"] for e in resolved if e.get("canonical_name")
    )


def build_retrieval_text(row: dict) -> str:
    parts = [
        f"Company: {row['company_name']}",
        f"Section: {row['heading']}",
        f"Category: {row['section_category']}",
        f"Resolved entities: {row['resolved_entity_names'] or 'none'}",
        f"Table summaries: {row['table_summaries'] or 'none'}",
        f"Text: {row['chunk_text'][:4000]}",
    ]
    return "\n\n".join(parts)


def make_record_id(source_path: str, section_heading: str) -> str:
    raw = f"{source_path}|{section_heading}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── core ──────────────────────────────────────────────────────────────────────

def standardize_records(records: list[dict], output_path: Path) -> pd.DataFrame:
    """Convert a list of pipeline records to a standardized Parquet file.

    Also returns the resulting DataFrame so callers can inspect it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_fields = 0
    chunk_index_by_doc: dict[str, int] = {}

    for record in records:
        required = {"section_heading", "section_category", "section_text"}
        if not required.issubset(record):
            missing = required - record.keys()
            print(f"  WARNING: missing fields {missing} in document {record.get('document_id', '?')}")
            missing_fields += 1
            continue

        record.setdefault("entities", [])
        record.setdefault("resolved_entities", [])
        record.setdefault("tables", [])
        record.setdefault("page_numbers", [])
        record.setdefault("processing_warnings", [])

        doc_id = record.get("document_id", "")
        source_path = record.get("source_path", "")
        section_heading = record["section_heading"]

        chunk_index_by_doc.setdefault(doc_id, 0)
        chunk_index = chunk_index_by_doc[doc_id]
        chunk_index_by_doc[doc_id] += 1

        company_name, cik, filing_year = parse_source_path(source_path)
        tables_clean = [
            {k: v for k, v in t.items() if k != "raw_text"}
            for t in record["tables"]
        ]

        row = {
            "document_id":              doc_id,
            "record_id":                make_record_id(source_path, section_heading),
            "chunk_index":              chunk_index,
            "source_filename":          Path(source_path).name,
            "company_name":             company_name,
            "cik":                      cik,
            "filing_year":              filing_year,
            "heading":                  section_heading,
            "section_category":         record["section_category"],
            "page_numbers_json":        serialize(record["page_numbers"]),
            "chunk_text":               record["section_text"],
            "entities_json":            serialize(record["entities"]),
            "resolved_entities_json":   serialize(record["resolved_entities"]),
            "tables_json":              serialize(tables_clean),
            "table_summaries":          get_table_summaries(record["tables"]),
            "entity_names":             get_entity_names(record["entities"]),
            "resolved_entity_names":    get_resolved_names(record["resolved_entities"]),
            "has_entities":             bool(record["entities"]),
            "has_resolved_entities":    bool(record["resolved_entities"]),
            "has_tables":               bool(record["tables"]),
            "table_count":              len(record["tables"]),
            "processing_warnings_json": serialize(record["processing_warnings"]),
            "warning_count":            len(record["processing_warnings"]),
            "text_char_count":          len(record["section_text"]),
        }
        row["retrieval_text"] = build_retrieval_text(row)
        rows.append(row)

    df = pd.DataFrame(rows)

    assert df["record_id"].is_unique, "record_id is not unique"
    assert df["chunk_text"].notna().all(), "some rows have null chunk_text"
    assert df["retrieval_text"].notna().all(), "some rows have null retrieval_text"

    df.to_parquet(output_path, index=False)

    n_docs = df["document_id"].nunique()
    print(f"  Standardized {len(df)} chunks from {n_docs} documents → {output_path}")
    if missing_fields:
        print(f"  Skipped {missing_fields} records (missing required fields)")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize pipeline JSONL to Parquet.")
    parser.add_argument(
        "--input", type=Path,
        default=Path("outputs/intermediate/all_chunks_with_entities_resolved_tables.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/final/chunks.parquet"))
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {args.input}")
    df = standardize_records(records, args.output)

    print(f"\nDocuments processed:    {df['document_id'].nunique()}")
    print(f"Chunk records written:  {len(df)}")
    print(f"Chunks with tables:     {df['has_tables'].sum()}")
    print(f"Total tables:           {df['table_count'].sum()}")
    print(f"Output:                 {args.output}")


if __name__ == "__main__":
    main()
