"""Lightweight inspection utilities for interactive use in notebooks and CLI.

All functions print to stdout and return nothing (or a small pandas object).
"""

# %%
import random
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional
from html import escape

import pandas as pd

from src.constants import RANDOM_SEED, EVALUATION_DIR
from src.utils import read_jsonl, truncate_text

from src.constants import EVALUATION_DIR


# ── document-level ────────────────────────────────────────────────────────────

def show_document_inventory(documents: list[dict]) -> pd.DataFrame:
    """Print a summary table of loaded documents."""
    rows = []
    for d in documents:
        text = d.get("raw_text", "")
        rows.append({
            "document_id": d["document_id"],
            "filename": Path(d["source_path"]).name,
            "file_type": d["file_type"],
            "char_count": len(text),
            "line_count": text.count("\n"),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def preview_document(document: dict, n_chars: int = 1000) -> None:
    """Print the first n_chars of a document's raw text."""
    text = document.get("raw_text", "")
    print(f"=== {Path(document['source_path']).name} ===")
    print(text[:n_chars])
    print(f"\n... ({len(text)} total chars)")




def save_chunks_html(chunks, directory=EVALUATION_DIR/"section"):
    """
    Save chunks to an HTML file without displaying them in the notebook.
    """

    if not chunks:
        raise ValueError("chunks cannot be empty")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    source_path = chunks[0].get("source_path", "document")
    document_name = Path(source_path).stem

    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in document_name
    )

    output_path = directory / f"{safe_name}.html"

    cards = []

    for i, chunk in enumerate(chunks, start=1):
        cards.append(
            f"""
            <div class="chunk">
                <h3>Chunk {i}</h3>

                <p>
                    <strong>Heading:</strong>
                    {escape(str(chunk.get("section_heading", "—")))}
                </p>

                <p>
                    <strong>Category:</strong>
                    {escape(str(chunk.get("section_category", "—")))}
                </p>

                <pre>{escape(str(chunk.get("section_text", "")))}</pre>
            </div>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{escape(document_name)}</title>

        <style>
            body {{
                max-width: 1000px;
                margin: 40px auto;
                padding: 0 20px;
                font-family: Arial, sans-serif;
                background: #f5f5f5;
            }}

            .chunk {{
                margin: 16px 0;
                padding: 18px;
                border: 1px solid #ddd;
                border-radius: 10px;
                background: white;
            }}

            .document-summary {{
                margin: 16px 0 24px;
                padding: 18px;
                border: 1px solid #ddd;
                border-radius: 10px;
                background: white;
            }}

            .document-summary h2 {{
                margin-top: 0;
            }}

            pre {{
                padding: 14px;
                border-radius: 6px;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                background: #f7f7f7;
            }}
        </style>
    </head>

    <body>
        <h1>{escape(document_name)}</h1>
        {''.join(cards)}
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")

    print(f"Saved {len(chunks)} chunks to: {output_path}")

    return output_path


def save_chunks_html_detailed(
    chunks,
    directory=EVALUATION_DIR / "section",
    snippet_chars: int = 500,
    entity_preview_items: int = 10,
    table_preview_items: int = 5,
):
    """
    Save chunks to an HTML file with a text snippet and structured chunk data.
    """

    if not chunks:
        raise ValueError("chunks cannot be empty")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    source_path = chunks[0].get("source_path", "document")
    document_name = Path(source_path).stem

    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in document_name
    )

    output_path = directory / f"{safe_name}_detailed.html"

    def format_counts(counter: Counter) -> str:
        if not counter:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in counter.most_common())

    all_tables = [table for chunk in chunks for table in chunk.get("tables", [])]
    chunks_with_tables = sum(1 for chunk in chunks if chunk.get("tables"))
    document_table_overview = {
        "records": len(chunks),
        "records_with_tables": chunks_with_tables,
        "tables": len(all_tables),
        "summary_sources": Counter(t.get("summary_source", "unknown") or "unknown" for t in all_tables),
        "summary_cache": Counter(t.get("summary_cache_status", "") or "none" for t in all_tables),
        "table_types": Counter(t.get("table_type", "unknown") or "unknown" for t in all_tables),
        "llm_summarized": sum(1 for t in all_tables if t.get("summary_used_llm")),
    }

    def render_value(value, max_items: int | None = None):
        if value is None:
            return "—"
        if isinstance(value, dict):
            return json.dumps(value, indent=2, ensure_ascii=False)
        if isinstance(value, (list, tuple)):
            if max_items is None or len(value) <= max_items:
                return json.dumps(value, indent=2, ensure_ascii=False)

            preview = list(value[:max_items])
            omitted = len(value) - max_items
            preview.append({"...": f"{omitted} more item(s) omitted"})
            return json.dumps(preview, indent=2, ensure_ascii=False)
        return str(value)

    def render_table_summary(tables: list[dict]) -> str:
        if not tables:
            return "<p class=\"muted\">No tables detected.</p>"

        source_counts = Counter(t.get("summary_source", "unknown") or "unknown" for t in tables)
        llm_count = sum(1 for t in tables if t.get("summary_used_llm"))
        cache_counts = Counter(
            t.get("summary_cache_status", "") or "none"
            for t in tables
        )
        overview = (
            f"{len(tables)} table{'s' if len(tables) != 1 else ''}; "
            f"{llm_count} LLM summarized; "
            f"sources: {format_counts(source_counts)}; "
            f"cache: {format_counts(cache_counts)}"
        )

        rows = []
        for idx, table in enumerate(tables, start=1):
            columns = table.get("columns", []) or []
            parsed_rows = table.get("rows", []) or []
            raw_text = table.get("raw_text", "") or ""
            first_row = parsed_rows[0] if parsed_rows else {}
            first_row_preview = truncate_text(
                json.dumps(first_row, ensure_ascii=False) if first_row else "—",
                220,
            )
            columns_preview = ", ".join(str(c) for c in columns[:6])
            if len(columns) > 6:
                columns_preview += f", +{len(columns) - 6} more"
            rows.append(
                f"""
                <tr>
                    <td>{idx}</td>
                    <td>{escape(str(table.get("table_type", "—") or "—"))}</td>
                    <td>{len(parsed_rows)} × {len(columns)}</td>
                    <td>{escape(str(table.get("summary_source", "—") or "—"))}</td>
                    <td>{escape(str(table.get("summary_used_llm", False)))}</td>
                    <td>{escape(str(table.get("summary_cache_status", "—") or "—"))}</td>
                    <td>{escape(str(table.get("units", "—") or "—"))}</td>
                    <td>{escape(columns_preview or "—")}</td>
                    <td>{escape(str(table.get("summary", "—") or "—"))}</td>
                    <td>{escape(first_row_preview)}</td>
                    <td>{len(raw_text):,}</td>
                </tr>
                """
            )

        return f"""
        <p class="table-overview">{escape(overview)}</p>
        <div class="table-scroll">
            <table class="table-summary">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Type</th>
                        <th>Rows × Cols</th>
                        <th>Summary Source</th>
                        <th>Used LLM</th>
                        <th>Cache</th>
                        <th>Units</th>
                        <th>Columns</th>
                        <th>Summary</th>
                        <th>First Row Preview</th>
                        <th>Raw Chars</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    cards = []

    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("chunk_text") or chunk.get("section_text", "")
        snippet = truncate_text(str(text), snippet_chars)
        entities = render_value(chunk.get("entities", []), max_items=entity_preview_items)
        resolved_entities = render_value(chunk.get("resolved_entities", []), max_items=entity_preview_items)
        table_records = chunk.get("tables", [])
        table_summary = render_table_summary(table_records)
        tables = render_value(table_records, max_items=table_preview_items)
        warnings = render_value(chunk.get("processing_warnings", []))
        page_numbers = render_value(chunk.get("page_numbers", []))
        record_id = chunk.get("chunk_id") or f"{chunk.get('document_id', 'document')}#{i}"

        cards.append(
            f"""
            <div class="chunk">
                <h3>Chunk {i}</h3>

                <p>
                    <strong>Heading:</strong>
                    {escape(str(chunk.get("section_heading", "—")))}
                </p>

                <p>
                    <strong>Category:</strong>
                    {escape(str(chunk.get("section_category", "—")))}
                </p>

                <p>
                    <strong>Record ID:</strong>
                    {escape(str(record_id))}
                </p>

                <p>
                    <strong>Page Numbers:</strong>
                    {escape(str(page_numbers))}
                </p>

                <p>
                    <strong>Text Snippet:</strong>
                </p>
                <pre>{escape(snippet)}</pre>

                <p>
                    <strong>Entities:</strong>
                </p>
                <pre>{escape(entities)}</pre>

                <p>
                    <strong>Resolved Entities:</strong>
                </p>
                <pre>{escape(resolved_entities)}</pre>

                <p>
                    <strong>Table Summary:</strong>
                </p>
                {table_summary}

                <p>
                    <strong>Tables Raw JSON Preview:</strong>
                </p>
                <pre>{escape(tables)}</pre>

                <p>
                    <strong>Processing Warnings:</strong>
                </p>
                <pre>{escape(warnings)}</pre>
            </div>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{escape(document_name)}</title>

        <style>
            body {{
                max-width: 1100px;
                margin: 40px auto;
                padding: 0 20px;
                font-family: Arial, sans-serif;
                background: #f5f5f5;
            }}

            .chunk {{
                margin: 16px 0;
                padding: 18px;
                border: 1px solid #ddd;
                border-radius: 10px;
                background: white;
            }}

            pre {{
                padding: 14px;
                border-radius: 6px;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                background: #f7f7f7;
            }}

            .muted {{
                color: #666;
            }}

            .table-overview {{
                margin: 8px 0 10px;
                color: #333;
            }}

            .table-scroll {{
                overflow-x: auto;
                margin-bottom: 14px;
            }}

            .table-summary {{
                width: 100%;
                border-collapse: collapse;
                background: #fff;
                font-size: 13px;
            }}

            .table-summary th,
            .table-summary td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }}

            .table-summary th {{
                background: #f0f0f0;
                font-weight: 700;
            }}
        </style>
    </head>

    <body>
        <h1>{escape(document_name)}</h1>
        <section class="document-summary">
            <h2>Table Overview</h2>
            <p>
                <strong>Records:</strong> {document_table_overview["records"]} ·
                <strong>Records with tables:</strong> {document_table_overview["records_with_tables"]} ·
                <strong>Total tables:</strong> {document_table_overview["tables"]} ·
                <strong>LLM summarized:</strong> {document_table_overview["llm_summarized"]}
            </p>
            <p>
                <strong>Summary sources:</strong>
                {escape(format_counts(document_table_overview["summary_sources"]))}
            </p>
            <p>
                <strong>Summary cache:</strong>
                {escape(format_counts(document_table_overview["summary_cache"]))}
            </p>
            <p>
                <strong>Table types:</strong>
                {escape(format_counts(document_table_overview["table_types"]))}
            </p>
        </section>
        {''.join(cards)}
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")

    print(f"Saved {len(chunks)} detailed chunks to: {output_path}")

    return output_path






# ── section-level ─────────────────────────────────────────────────────────────

def show_sections(chunks: list[dict], doc_id: Optional[str] = None) -> None:
    """Print detected sections (unique heading + category) for a document."""
    seen: set[tuple] = set()
    for c in chunks:
        if doc_id and c["document_id"] != doc_id:
            continue
        key = (c["section_heading"], c["section_category"])
        if key not in seen:
            seen.add(key)
            print(f"  [{c['section_category']:35s}]  {c['section_heading'][:80]}")


# ── chunk-level ───────────────────────────────────────────────────────────────

def show_chunks(chunks: list[dict], n: int = 5, category: Optional[str] = None) -> None:
    """Print n sample chunks, optionally filtered by category."""
    filtered = [c for c in chunks if not category or c["section_category"] == category]
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(filtered, min(n, len(filtered)))
    for c in sample:
        print(f"\n--- chunk_id={c['chunk_id']}  [{c['section_category']}] ---")
        print(f"Heading: {c['section_heading'][:80]}")
        print(truncate_text(c["chunk_text"], 300))


# ── entity-level ──────────────────────────────────────────────────────────────

def show_entities(chunks: list[dict], n: int = 10) -> None:
    """Print the most frequent entity mentions across all chunks."""
    counter: Counter = Counter()
    for c in chunks:
        for e in c.get("entities", []):
            counter[(e["text"], e["entity_type"])] += 1
    print(f"{'Count':>6}  {'Type':15}  Entity")
    print("-" * 50)
    for (text, etype), cnt in counter.most_common(n):
        print(f"{cnt:>6}  {etype:15}  {text}")


def show_entities_by_type(chunks: list[dict], entity_type: str, n: int = 10) -> None:
    """Print the most frequent entity mentions of a specific type across all chunks."""
    counter: Counter = Counter()
    for c in chunks:
        for e in c.get("entities", []):
            counter[(e["text"], e["entity_type"])] += 1
    print(f"{'Count':>6}  {'Type':15}  Entity")
    print("-" * 50)
    for (text, etype), cnt in counter.most_common(n):
        print(f"{cnt:>6}  {etype:15}  {text}")


def show_resolved_groups(resolution_map: list[dict], n: int = 10) -> None:
    """Print top canonical entity groups."""
    print(f"{'Count':>6}  {'Type':10}  {'Method':20}  Canonical → variants")
    print("-" * 80)
    for g in resolution_map[:n]:
        variants = ", ".join(g["variants"][:4])
        extra = f" (+{len(g['variants'])-4} more)" if len(g["variants"]) > 4 else ""
        print(
            f"{g['mention_count']:>6}  {g['entity_type']:10}  "
            f"{g['resolution_method']:20}  {g['canonical_name']} → [{variants}{extra}]"
        )


# ── table-level ───────────────────────────────────────────────────────────────

def show_tables(chunks: list[dict], n: int = 3) -> None:
    """Print raw and parsed representations of up to n tables."""
    count = 0
    for c in chunks:
        for t in c.get("tables", []):
            if count >= n:
                return
            print(f"\n=== Table in '{c['section_heading'][:60]}' ===")
            print("RAW:")
            print(truncate_text(t.get("raw_text", ""), 400))
            print("\nPARSED columns:", t.get("columns"))
            print("PARSED rows (first 3):", t.get("rows", [])[:3])
            print("SUMMARY:", t.get("summary", ""))
            count += 1


# ── warning inspection ────────────────────────────────────────────────────────

def show_warnings(chunks: list[dict]) -> None:
    """Print chunks that contain processing warnings."""
    warned = [c for c in chunks if c.get("processing_warnings")]
    print(f"{len(warned)} chunks with warnings:")
    for c in warned:
        print(f"  {c['chunk_id']}: {c['processing_warnings']}")


# ── PDF-specific ──────────────────────────────────────────────────────────────

def show_pdf_quality(documents: list[dict]) -> None:
    """Print per-page quality scores for PDF documents."""
    for d in documents:
        if d["file_type"] != "pdf":
            continue
        pages = d["metadata"].get("pages", [])
        print(f"\n{Path(d['source_path']).name} ({len(pages)} pages):")
        for p in pages:
            flag = "⚠" if p["warnings"] else "✓"
            print(
                f"  Page {p['page_number']:3d}  score={p['quality_score']:.2f} "
                f"{flag}  {p['warnings']}"
            )


# ── from saved JSONL ──────────────────────────────────────────────────────────

def load_and_show(jsonl_path: Path, show: str = "sections", n: int = 10) -> list[dict]:
    """Load chunks from JSONL and display the requested view."""
    chunks = read_jsonl(jsonl_path)
    if show == "sections":
        show_sections(chunks)
    elif show == "entities":
        show_entities(chunks, n=n)
    elif show == "tables":
        show_tables(chunks, n=n)
    elif show == "warnings":
        show_warnings(chunks)
    return chunks

# ── Manual inspection ──────────────────────────────────────────────────────────
