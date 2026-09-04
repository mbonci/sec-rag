"""Validation script: summary statistics, histograms, and outlier detection.

Runs on any pipeline JSONL output at any stage. Produces a terminal report
and saves a multi-panel figure as a PNG next to the input file.

Usage:
    python scripts/validate_outputs.py
    python scripts/validate_outputs.py --input outputs/intermediate/all_chunks_with_entities.jsonl
    python scripts/validate_outputs.py --input outputs/final/chunks.jsonl --no-plot
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import OUTPUT_JSONL, SECTION_CATEGORIES

# ── terminal formatting ───────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"


def _h(title: str) -> None:
    width = 70
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠  {msg}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓  {msg}{RESET}")


def _err(msg: str) -> None:
    print(f"  {RED}✗  {msg}{RESET}")


def _bar(label: str, count: int, total: int, width: int = 30) -> str:
    filled = int(width * count / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * count / total if total else 0
    return f"  {label:<40s} {bar}  {count:>5d}  ({pct:5.1f}%)"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "0/0"


def _stats(values: list) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min":  s[0],
        "max":  s[-1],
        "mean": sum(s) / n,
        "p25":  s[int(n * 0.25)],
        "p50":  s[n // 2],
        "p75":  s[int(n * 0.75)],
        "p95":  s[int(n * 0.95)],
    }


def _doc_label(doc_id: str, chunks: list[dict]) -> str:
    """Return a short human-readable label for a doc_id."""
    for c in chunks:
        if c["document_id"] == doc_id:
            path = c.get("source_path", "")
            stem = Path(path).stem if path else doc_id
            return f"{stem[:50]}  [{doc_id}]"
    return doc_id


def _iqr_bounds(values: list, k: float = 1.5) -> tuple[float, float]:
    s = sorted(values)
    n = len(s)
    q1, q3 = s[int(n * 0.25)], s[int(n * 0.75)]
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# ── task validators ───────────────────────────────────────────────────────────

def validate_sections(chunks: list[dict]) -> tuple[int, dict]:
    """Task 1 — section extraction, chunking, and categorization.
    Returns (issue_count, plot_data).
    """
    _h("TASK 1 · Section Extraction & Categorization")
    issues = 0

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["document_id"]].append(c)

    n_docs = len(by_doc)
    chunks_per_doc = {did: len(v) for did, v in by_doc.items()}
    cpd_vals = list(chunks_per_doc.values())
    st = _stats(cpd_vals)

    print(f"\n  Documents: {n_docs}   Total chunks: {len(chunks)}")
    print(f"  Chunks/doc — min {st['min']:.0f}  mean {st['mean']:.1f}  "
          f"p50 {st['p50']:.0f}  p95 {st['p95']:.0f}  max {st['max']:.0f}")

    # Category distribution
    print("\n  Category distribution:")
    cat_counts = Counter(c["section_category"] for c in chunks)
    for cat in SECTION_CATEGORIES:
        print(_bar(cat, cat_counts.get(cat, 0), len(chunks)))
    unknown = {k: v for k, v in cat_counts.items() if k not in SECTION_CATEGORIES}
    for cat, cnt in unknown.items():
        _warn(f"Unknown category '{cat}': {cnt} chunks")
        issues += 1

    # Preamble coverage
    preamble_docs = sum(
        1 for v in by_doc.values()
        if any(c.get("section_heading") == "Preamble" for c in v)
    )
    print(f"\n  Preamble present in {_pct(preamble_docs, n_docs)} documents")

    text_key = "section_text"
    sizes = [len(c.get(text_key, "")) for c in chunks]
    ss = _stats(sizes)
    print(f"\n  Section text length — min {ss['min']:.0f}  mean {ss['mean']:.0f}  "
          f"p50 {ss['p50']:.0f}  p95 {ss['p95']:.0f}  max {ss['max']:.0f}")

    empty = [c for c in chunks if not c.get(text_key, "").strip()]
    if empty:
        _err(f"{len(empty)} chunks have empty text")
        issues += len(empty)
    else:
        _ok("No empty chunk bodies")

    tiny = [(c["document_id"], len(c.get(text_key, ""))) for c in chunks if 0 < len(c.get(text_key, "")) < 100]
    if tiny:
        _warn(f"{len(tiny)} chunks shorter than 100 chars")
        issues += len(tiny)

    # Outliers: chunks per document (IQR)
    _, high_cpd = _iqr_bounds(cpd_vals)
    low_cpd = 2
    cpd_outliers_low  = [(did, n) for did, n in chunks_per_doc.items() if n <= low_cpd]
    cpd_outliers_high = [(did, n) for did, n in chunks_per_doc.items() if n > high_cpd]

    if cpd_outliers_low:
        print(f"\n  {YELLOW}Outlier documents — very few chunks (<= {low_cpd}):{RESET}")
        for did, n in sorted(cpd_outliers_low, key=lambda x: x[1]):
            print(f"    {n} chunks  →  {_doc_label(did, chunks)}")
            issues += 1
    if cpd_outliers_high:
        print(f"\n  {YELLOW}Outlier documents — unusually many chunks (> {high_cpd:.0f}, IQR×1.5):{RESET}")
        for did, n in sorted(cpd_outliers_high, key=lambda x: -x[1]):
            print(f"    {n} chunks  →  {_doc_label(did, chunks)}")

    # Outliers: chunk text length (IQR on non-empty)
    nonempty_sizes = [s for s in sizes if s > 0]
    _, high_size = _iqr_bounds(nonempty_sizes)
    size_outliers = [
        (c["document_id"], len(c.get(text_key, "")))
        for c in chunks if len(c.get(text_key, "")) > high_size
    ]
    if size_outliers:
        top = sorted(set(size_outliers), key=lambda x: -x[1])[:5]
        print(f"\n  {YELLOW}Outlier chunks — very long text (> {high_size:.0f} chars, IQR×1.5):{RESET}")
        for did, sz in top:
            print(f"    {sz:>8,} chars  →  {_doc_label(did, chunks)}")

    plot_data = {
        "chunk_sizes":     sizes,
        "chunks_per_doc":  cpd_vals,
        "cpd_by_doc":      chunks_per_doc,
        "cat_counts":      cat_counts,
        "text_key":        text_key,
        "high_cpd":        high_cpd,
        "high_size":       high_size,
    }
    return issues, plot_data


def validate_entities(chunks: list[dict]) -> tuple[int, dict]:
    """Task 2 — entity extraction."""
    _h("TASK 2 · Entity Extraction")
    issues = 0

    all_entities = [e for c in chunks for e in c.get("entities", [])]
    chunks_with = sum(1 for c in chunks if c.get("entities"))
    print(f"\n  Total entities extracted: {len(all_entities)}")
    print(f"  Chunks with ≥1 entity:   {_pct(chunks_with, len(chunks))}")

    if not all_entities:
        _err("No entities found — entity extraction may have failed")
        return issues + 1, {}

    type_counts = Counter(e.get("entity_type", "UNKNOWN") for e in all_entities)
    print("\n  Entity type distribution:")
    for label, cnt in type_counts.most_common():
        print(_bar(label, cnt, len(all_entities)))

    ent_per_chunk = [len(c.get("entities", [])) for c in chunks]
    st = _stats(ent_per_chunk)
    print(f"\n  Entities/chunk — mean {st['mean']:.1f}  p50 {st['p50']:.0f}  "
          f"p95 {st['p95']:.0f}  max {st['max']:.0f}")

    _, high_ent = _iqr_bounds([x for x in ent_per_chunk if x > 0])
    noisy = [
        (c.get("document_id", "?"), c.get("section_heading", "?"), len(c["entities"]))
        for c in chunks if len(c.get("entities", [])) > high_ent
    ]
    if noisy:
        print(f"\n  {YELLOW}Outlier chunks — very high entity count (> {high_ent:.0f}, IQR×1.5):{RESET}")
        for did, cid, n in sorted(noisy, key=lambda x: -x[2])[:5]:
            print(f"    {n:>4} entities  →  chunk {cid}  doc {_doc_label(did, chunks)}")
            issues += 1

    monetary = [e for e in all_entities if e.get("entity_type", "").lower() in
                ("monetary", "money", "monetary_value")]
    print(f"\n  Monetary entities: {len(monetary)} ({100*len(monetary)/max(len(all_entities),1):.1f}% of all)")
    if not monetary:
        _warn("No monetary entities found — check regex patterns")
        issues += 1

    # Top 10 per type
    for etype in ("company", "person", "monetary_value"):
        typed = [e["text"] for e in all_entities if e.get("entity_type") == etype]
        if not typed:
            continue
        top = Counter(typed).most_common(10)
        print(f"\n  Top 10 {etype} entities:")
        for name, cnt in top:
            print(f"    {cnt:>5}×  {name}")

    plot_data = {
        "type_counts":    type_counts,
        "ent_per_chunk":  ent_per_chunk,
    }
    return issues, plot_data


def validate_entity_resolution(chunks: list[dict]) -> tuple[int, dict]:
    """Task 3 — entity resolution."""
    _h("TASK 3 · Entity Resolution")
    issues = 0

    # ── metric definitions ────────────────────────────────────────────────────
    print(f"\n  {BOLD}Metric definitions:{RESET}")
    print("    Raw entities       — all extracted mentions (company + person + monetary_value)")
    print("    Resolvable         — company + person mentions only (candidates for grouping)")
    print("    Resolved entries   — resolvable mentions assigned a canonical name")
    print("    Coverage           — resolved / resolvable")
    print("    Unique canonicals  — distinct representative names after grouping variants")
    print("    Singletons         — canonicals that appear in only one chunk (no cross-chunk merge)")

    # ── core counts ──────────────────────────────────────────────────────────
    raw_total = sum(len(c.get("entities", [])) for c in chunks)
    res_total = sum(len(c.get("resolved_entities", [])) for c in chunks)
    res_chunks = sum(1 for c in chunks if c.get("resolved_entities"))

    _RESOLVABLE = {"company", "person"}
    resolvable = sum(
        1 for c in chunks for e in c.get("entities", [])
        if e.get("entity_type", "").lower() in _RESOLVABLE
    )
    coverage = res_total / resolvable if resolvable else 0

    print(f"\n  {'Metric':<40} {'Value':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Raw entities':<40} {raw_total:>10,}")
    print(f"  {'Resolvable (company + person)':<40} {resolvable:>10,}")
    print(f"  {'Resolved entries':<40} {res_total:>10,}")
    print(f"  {'Coverage':<40} {coverage:>10.1%}")

    if raw_total == 0:
        _err("No raw entities to resolve")
        return issues + 1, {}

    if coverage < 0.5:
        _warn("Fewer than 50% of resolvable entities have resolved forms")
        issues += 1

    # ── canonical name stats ──────────────────────────────────────────────────
    canonical_counts: Counter = Counter()
    for c in chunks:
        for e in c.get("resolved_entities", []):
            name = e.get("canonical_name") or e.get("text", "")
            if name:
                canonical_counts[name] += 1

    singletons = sum(1 for v in canonical_counts.values() if v == 1)
    pct_single = 100 * singletons / len(canonical_counts) if canonical_counts else 0

    print(f"  {'Unique canonical names':<40} {len(canonical_counts):>10,}")
    print(f"  {'Chunks with resolved entities':<40} {_pct(res_chunks, len(chunks)):>10}")
    print(f"  {'Singleton canonicals':<40} {singletons:>10,}  ({pct_single:.1f}%)")

    if pct_single > 70:
        _warn("High singleton rate — resolution threshold may be too strict or data too sparse")
        issues += 1

    if canonical_counts:
        print(f"\n  Top 10 most-referenced canonical entities:")
        for name, cnt in canonical_counts.most_common(10):
            print(f"    {cnt:>5}×  {name}")

    # ── resolution method breakdown ───────────────────────────────────────────
    all_resolved = [e for c in chunks for e in c.get("resolved_entities", [])]

    print(f"\n  {BOLD}Resolution method breakdown:{RESET}")
    print(f"    {'Method':<22} {'Count':>7}  {'%':>6}  Definition")
    print(f"    {'-'*75}")

    _METHOD_DEFS = {
        "canonical":        "mention is the chosen representative — no merge needed",
        "normalized_exact": "matched after stripping legal suffixes and lowercasing",
        "strict_fuzzy":     "fuzzy score ≥ threshold on both ratio and token_sort",
        "person_name":      "shared first + last name (person-specific rule)",
        "person_initials":  "matching initials + shared last name",
        "acronym":          "short all-caps form matched to long-form word initials",
        "cluster_link":     "transitively connected — no direct merge rule fired",
        "singleton":        "only one mention in document; kept as-is",
        "mixed":            "group merged via more than one method",
    }

    method_counts = Counter(e.get("resolution_method", "unknown") for e in all_resolved)
    for method, cnt in method_counts.most_common():
        pct = 100 * cnt / res_total if res_total else 0
        defn = _METHOD_DEFS.get(method, "")
        print(f"    {method:<22} {cnt:>7,}  {pct:>5.1f}%  {defn}")

    cluster_links = [e for e in all_resolved if e.get("resolution_method") == "cluster_link"]
    suspicious = [e for e in cluster_links if (e.get("resolution_score") or 100) < 50]
    if suspicious:
        _warn(f"{len(suspicious)} cluster_link entries with score < 50 (possible false merges)")
        issues += 1

    # ── resolution score distribution ────────────────────────────────────────
    non_canonical_scores = [
        e["resolution_score"] for e in all_resolved
        if e.get("resolution_score") is not None
        and e.get("resolution_method") not in ("canonical", "singleton")
    ]
    if non_canonical_scores:
        ss = _stats(non_canonical_scores)
        print(f"\n  {BOLD}Resolution score distribution{RESET} (excludes canonical/singleton entries):")
        print(f"    min {ss['min']:.0f}  p25 {ss.get('p25', 0):.0f}  "
              f"p50 {ss['p50']:.0f}  p75 {ss.get('p75', 0):.0f}  max {ss['max']:.0f}")
        print(f"    Score = 100 → exact / acronym match")
        print(f"    Score 90–99 → fuzzy match above threshold")
        print(f"    Score  < 50 → cluster_link with weak direct evidence")
        low_score = sum(1 for s in non_canonical_scores if s < 50)
        if low_score:
            _warn(f"{low_score} non-canonical entries with score < 50")
            issues += 1

    # ── per-method examples ───────────────────────────────────────────────────
    print(f"\n  {BOLD}One example per resolution method:{RESET}")
    seen_methods: set[str] = set()
    for e in all_resolved:
        m = e.get("resolution_method", "unknown")
        if m in seen_methods:
            continue
        seen_methods.add(m)
        matched = e.get("matched_text", "")[:35]
        canonical = e.get("canonical_name", "")[:35]
        score = e.get("resolution_score", "–")
        same = "  (same)" if matched == canonical else ""
        print(f"    {m:<22}  {matched!r:<37} → {canonical!r}  [score {score}]{same}")

    # collect one example per method for the markdown report
    examples_by_method: dict[str, dict] = {}
    for e in all_resolved:
        m = e.get("resolution_method", "unknown")
        if m not in examples_by_method:
            examples_by_method[m] = e

    return issues, {
        "canonical_counts": canonical_counts,
        "method_counts": method_counts,
        "non_canonical_scores": non_canonical_scores,
        "examples_by_method": examples_by_method,
        "method_defs": _METHOD_DEFS,
    }


def _load_llm_cost_stats() -> dict:
    """Read cumulative stats from the cache file and per-call stats from the metrics CSV."""
    import csv as _csv, json as _json
    repo = Path(__file__).parent.parent

    # ── cumulative cache stats ────────────────────────────────────────────────
    cache_path = repo / "outputs" / "cache" / "table_llm_summary_cache.json"
    entries: list[dict] = []
    if cache_path.exists():
        try:
            entries = list(_json.loads(cache_path.read_text()).values())
        except Exception:
            pass
    summary_entries = [e for e in entries if "_infer" not in e.get("cache_version", "")]
    infer_entries   = [e for e in entries if "_infer"     in e.get("cache_version", "")]

    # ── per-run call log ──────────────────────────────────────────────────────
    metrics_path = repo / "outputs" / "costs" / "table_llm_summary_metrics.csv"
    rows: list[dict] = []
    if metrics_path.exists():
        try:
            rows = list(_csv.DictReader(metrics_path.open()))
        except Exception:
            pass
    fresh_calls  = [r for r in rows if r.get("status", "").startswith("ok")]
    cache_hits   = [r for r in rows if r.get("status") == "cache_hit"]
    errors       = [r for r in rows if r.get("status") == "error"]
    total_logged = len(rows)
    hit_rate     = len(cache_hits) / total_logged if total_logged else 0.0

    return {
        # cumulative (from cache file)
        "total_entries":      len(entries),
        "summary_entries":    len(summary_entries),
        "infer_entries":      len(infer_entries),
        "prompt_tokens":      sum(e.get("prompt_tokens", 0) for e in entries),
        "completion_tokens":  sum(e.get("completion_tokens", 0) for e in entries),
        "total_tokens":       sum(e.get("total_tokens", 0) for e in entries),
        "estimated_cost_usd": sum(e.get("estimated_cost_usd", 0.0) for e in entries),
        # per-run (from metrics CSV)
        "run_fresh_calls":    len(fresh_calls),
        "run_cache_hits":     len(cache_hits),
        "run_errors":         len(errors),
        "run_total_logged":   total_logged,
        "run_hit_rate":       hit_rate,
    }


def validate_tables(chunks: list[dict]) -> tuple[int, dict]:
    """Task 4 — table detection and parsing."""
    _h("TASK 4 · Table Detection & Parsing")
    issues = 0

    chunks_with = [c for c in chunks if c.get("tables")]
    all_tables  = [t for c in chunks for t in c.get("tables", [])]
    print(f"\n  Chunks with ≥1 table: {_pct(len(chunks_with), len(chunks))}")
    print(f"  Total tables found:   {len(all_tables)}")

    if not all_tables:
        _warn("No tables detected — check table_parser patterns")
        return issues + 1, {}

    # ── row / column stats ────────────────────────────────────────────────────
    row_counts = [len(t.get("rows", [])) for t in all_tables]
    col_counts = [len(t.get("columns", [])) for t in all_tables]
    rs, cs = _stats(row_counts), _stats(col_counts)
    print(f"\n  Rows/table  — min {rs['min']:.0f}  mean {rs['mean']:.1f}  "
          f"p50 {rs['p50']:.0f}  p95 {rs['p95']:.0f}  max {rs['max']:.0f}")
    print(f"  Cols/table  — min {cs['min']:.0f}  mean {cs['mean']:.1f}  "
          f"p50 {cs['p50']:.0f}  max {cs['max']:.0f}")

    # ── column detection quality ──────────────────────────────────────────────
    with_cols = sum(1 for t in all_tables if t.get("columns"))
    with_named = sum(
        1 for t in all_tables
        if any(not c.startswith("Col") for c in t.get("columns", []) if c != "Item")
    )
    structured = sum(
        1 for t in all_tables
        if t.get("rows") and isinstance(t["rows"][0], dict)
    )
    print(f"\n  Column detection rate:     {_pct(with_cols, len(all_tables))}  (have ≥1 named column)")
    print(f"  Named (non-generic) cols:  {_pct(with_named, len(all_tables))}")
    print(f"  Rows structured as dicts:  {_pct(structured, len(all_tables))}")
    if with_cols < len(all_tables) * 0.5:
        _warn("Column detection rate below 50% — many tables lack named columns")
        issues += 1

    # ── table type breakdown ──────────────────────────────────────────────────
    type_counts: Counter = Counter(t.get("table_type", "unknown") for t in all_tables)
    print(f"\n  Table type breakdown:")
    for ttype, cnt in type_counts.most_common():
        print(f"    {ttype:<20} {cnt:>5}  ({100*cnt/len(all_tables):.1f}%)")

    # ── by section category ───────────────────────────────────────────────────
    cat_tables: dict[str, list] = {}
    for c in chunks:
        cat = c.get("section_category", "other")
        cat_tables.setdefault(cat, []).extend(c.get("tables", []))

    print(f"\n  Tables by section category:")
    print(f"    {'Category':<35} {'Tables':>7}  {'Chunks w/ table':>15}  {'Avg rows':>8}  {'Col detect':>10}")
    print(f"    {'-'*80}")
    for cat in sorted(cat_tables, key=lambda c: -len(cat_tables[c])):
        tbls = cat_tables[cat]
        cat_chunks = [c for c in chunks if c.get("section_category") == cat]
        cat_with = sum(1 for c in cat_chunks if c.get("tables"))
        avg_rows = sum(len(t.get("rows", [])) for t in tbls) / len(tbls) if tbls else 0
        col_rate = sum(1 for t in tbls if t.get("columns")) / len(tbls) if tbls else 0
        print(f"    {cat:<35} {len(tbls):>7}  "
              f"{_pct(cat_with, len(cat_chunks)):>15}  "
              f"{avg_rows:>8.1f}  "
              f"{col_rate:>9.0%}")

    # ── summary quality ───────────────────────────────────────────────────────
    with_summary = sum(1 for t in all_tables if t.get("summary", "").strip())
    llm_summary  = sum(1 for t in all_tables if t.get("summary_used_llm"))
    summary_lens = [len(t.get("summary", "")) for t in all_tables if t.get("summary")]
    sl = _stats(summary_lens) if summary_lens else {}
    print(f"\n  Summaries present:     {_pct(with_summary, len(all_tables))}")
    print(f"  LLM summaries:         {_pct(llm_summary, len(all_tables))}")
    if sl:
        print(f"  Summary length chars — mean {sl['mean']:.0f}  p50 {sl['p50']:.0f}  max {sl['max']:.0f}")

    # ── document coverage ─────────────────────────────────────────────────────
    by_doc: Counter = Counter()
    for c in chunks:
        by_doc[c["document_id"]] += len(c.get("tables", []))
    docs_with = sum(1 for v in by_doc.values() if v > 0)
    print(f"\n  Documents with ≥1 table: {_pct(docs_with, len(by_doc))}")
    if docs_with < len(by_doc) * 0.5:
        _warn("Fewer than half of documents have any detected tables")
        issues += 1

    # ── outliers ──────────────────────────────────────────────────────────────
    _, high_t = _iqr_bounds([len(c.get("tables", [])) for c in chunks if c.get("tables")])
    noisy = [
        (c["document_id"], c.get("section_heading", "?"), len(c["tables"]))
        for c in chunks if len(c.get("tables", [])) > high_t
    ]
    if noisy:
        print(f"\n  {YELLOW}Outlier chunks — very many tables (> {high_t:.0f}):{RESET}")
        for did, cid, n in sorted(noisy, key=lambda x: -x[2])[:5]:
            print(f"    {n} tables  →  chunk {cid[:70]}  doc {_doc_label(did, chunks)}")

    # ── LLM cost summary ──────────────────────────────────────────────────────
    cost_stats = _load_llm_cost_stats()
    if cost_stats:
        print(f"\n  {BOLD}LLM cost summary:{RESET}")
        if cost_stats.get("run_total_logged"):
            hit_pct = 100 * cost_stats["run_hit_rate"]
            print(f"    {BOLD}Last run:{RESET}")
            print(f"      {'Fresh API calls':<33} {cost_stats['run_fresh_calls']:>8,}")
            print(f"      {'Cache hits':<33} {cost_stats['run_cache_hits']:>8,}  ({hit_pct:.1f}%)")
            if cost_stats["run_errors"]:
                print(f"      {'Errors':<33} {cost_stats['run_errors']:>8,}")
        print(f"    {BOLD}Cumulative (all runs):{RESET}")
        print(f"      {'Unique API calls':<33} {cost_stats['total_entries']:>8,}")
        print(f"        {'→ named-column summaries':<31} {cost_stats['summary_entries']:>8,}")
        print(f"        {'→ generic-column inference':<31} {cost_stats['infer_entries']:>8,}")
        print(f"      {'Prompt tokens':<33} {cost_stats['prompt_tokens']:>8,}")
        print(f"      {'Completion tokens':<33} {cost_stats['completion_tokens']:>8,}")
        print(f"      {'Total tokens':<33} {cost_stats['total_tokens']:>8,}")
        print(f"      {'Estimated cost':<33} ${cost_stats['estimated_cost_usd']:>8.4f}")

    return issues, {
        "row_counts": row_counts,
        "col_counts": col_counts,
        "cat_tables": cat_tables,
        "type_counts": type_counts,
        "with_cols": with_cols,
        "structured": structured,
        "summary_lens": summary_lens,
        "cost_stats": cost_stats,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def build_report(
    chunks: list[dict],
    sec_data: dict,
    ent_data: dict,
    tbl_data: dict,
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    STYLE = {
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "font.size":          10,
    }
    plt.rcParams.update(STYLE)
    FIG_COLOR = "#f8f9fa"
    BAR_COLOR = "#4C72B0"
    OUT_COLOR = "#DD4444"

    fig = plt.figure(figsize=(18, 26), facecolor=FIG_COLOR)
    fig.suptitle(
        f"Pipeline Validation Report\n{output_path.stem}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.96, bottom=0.04,
                          height_ratios=[1, 1, 1, 0.9])

    # ── 1. Chunk text length histogram ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    text_key = sec_data.get("text_key", "section_text")
    sizes = sec_data["chunk_sizes"]
    high_size = sec_data["high_size"]

    normal = [s for s in sizes if s <= high_size and s > 0]
    outlier = [s for s in sizes if s > high_size]

    ax1.hist(normal, bins=40, color=BAR_COLOR, alpha=0.85, edgecolor="white", linewidth=0.4)
    if outlier:
        ax1.hist(outlier, bins=10, color=OUT_COLOR, alpha=0.85, edgecolor="white",
                 linewidth=0.4, label=f"Outliers >IQR×1.5 (n={len(outlier)})")
        ax1.legend(fontsize=8)
    ax1.axvline(sorted(sizes)[len(sizes)//2], color="navy", linestyle="--",
                linewidth=1.2, label=f"Median")
    ax1.set_xlabel("Chunk text length (chars)", labelpad=6)
    ax1.set_ylabel("Number of chunks")
    ax1.set_title("Chunk Text Length Distribution")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── 2. Chunks per document histogram ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cpd = sec_data["chunks_per_doc"]
    high_cpd = sec_data["high_cpd"]

    normal_cpd  = [v for v in cpd if v <= high_cpd]
    outlier_cpd = [v for v in cpd if v > high_cpd]

    ax2.hist(normal_cpd, bins=max(1, len(set(normal_cpd))), color=BAR_COLOR,
             alpha=0.85, edgecolor="white", linewidth=0.4)
    if outlier_cpd:
        ax2.hist(outlier_cpd, bins=max(1, len(set(outlier_cpd))),
                 color=OUT_COLOR, alpha=0.85, edgecolor="white", linewidth=0.4,
                 label=f"Outliers >IQR×1.5 (n={len(outlier_cpd)})")
        ax2.legend(fontsize=8)
    ax2.axvline(sorted(cpd)[len(cpd)//2], color="navy", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Chunks per document")
    ax2.set_ylabel("Number of documents")
    ax2.set_title("Chunks per Document Distribution")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── 3. Section category distribution (horizontal bar) ───────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cat_counts = sec_data["cat_counts"]
    cats = SECTION_CATEGORIES + [k for k in cat_counts if k not in SECTION_CATEGORIES]
    cat_vals = [cat_counts.get(c, 0) for c in cats]
    y_pos = range(len(cats))
    bars = ax3.barh(list(y_pos), cat_vals, color=BAR_COLOR, alpha=0.85, edgecolor="white")
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels([c.replace("_", " ") for c in cats], fontsize=8)
    ax3.set_xlabel("Number of chunks")
    ax3.set_title("Chunks by Section Category")
    for bar, val in zip(bars, cat_vals):
        if val:
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", fontsize=7)

    # ── 4. Entity type distribution (pie) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if ent_data.get("type_counts"):
        tc = ent_data["type_counts"]
        labels = [f"{k}\n({v:,})" for k, v in tc.most_common()]
        sizes_pie = [v for _, v in tc.most_common()]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"][:len(labels)]
        ax4.pie(sizes_pie, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=140, textprops={"fontsize": 8})
        ax4.set_title("Entity Type Distribution")
    else:
        ax4.text(0.5, 0.5, "No entity data", ha="center", va="center")
        ax4.set_title("Entity Type Distribution")

    # ── 5. Entities per chunk histogram ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if ent_data.get("ent_per_chunk"):
        epc = ent_data["ent_per_chunk"]
        _, high_epc = _iqr_bounds([x for x in epc if x > 0] or [0])
        normal_epc  = [v for v in epc if v <= high_epc]
        outlier_epc = [v for v in epc if v > high_epc]
        ax5.hist(normal_epc, bins=30, color=BAR_COLOR, alpha=0.85, edgecolor="white", linewidth=0.4)
        if outlier_epc:
            ax5.hist(outlier_epc, bins=10, color=OUT_COLOR, alpha=0.85, edgecolor="white",
                     linewidth=0.4, label=f"Outliers (n={len(outlier_epc)})")
            ax5.legend(fontsize=8)
        ax5.set_xlabel("Number of entities per chunk")
        ax5.set_ylabel("Number of chunks")
        ax5.set_title("Entities per Chunk Distribution")
    else:
        ax5.text(0.5, 0.5, "No entity data", ha="center", va="center")
        ax5.set_title("Entities per Chunk")

    # ── 6. Chunks per document — top 20 ranked bar ──────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    cpd_by_doc = sec_data["cpd_by_doc"]
    ranked = sorted(cpd_by_doc.items(), key=lambda x: -x[1])[:20]
    labels_6 = [did[:8] for did, _ in ranked]
    vals_6   = [v for _, v in ranked]
    colors_6 = [OUT_COLOR if v > high_cpd else BAR_COLOR for v in vals_6]
    ax6.barh(range(len(ranked)), vals_6, color=colors_6, alpha=0.85, edgecolor="white")
    ax6.set_yticks(range(len(ranked)))
    ax6.set_yticklabels(labels_6, fontsize=7, family="monospace")
    ax6.axvline(high_cpd, color=OUT_COLOR, linestyle="--", linewidth=1, alpha=0.7,
                label=f"IQR×1.5 threshold ({high_cpd:.0f})")
    ax6.legend(fontsize=7)
    ax6.set_xlabel("Number of chunks")
    ax6.set_title("Top 20 Documents by Chunk Count\n(red = outlier)")
    ax6.invert_yaxis()

    # ── 7. Table rows per table histogram ───────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    if tbl_data.get("row_counts"):
        ax7.hist(tbl_data["row_counts"], bins=30, color=BAR_COLOR, alpha=0.85,
                 edgecolor="white", linewidth=0.4)
        ax7.set_xlabel("Rows per table")
        ax7.set_ylabel("Number of tables")
        ax7.set_title("Table Size Distribution (Rows)")
    else:
        ax7.text(0.5, 0.5, "No table data", ha="center", va="center", transform=ax7.transAxes)
        ax7.set_title("Table Size Distribution (Rows)")

    # ── 8. Tables per document bar ───────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    tbl_per_doc: Counter = Counter()
    for c in chunks:
        tbl_per_doc[c["document_id"]] += len(c.get("tables", []))
    if tbl_per_doc:
        tpd_vals = sorted(tbl_per_doc.values(), reverse=True)
        ax8.bar(range(len(tpd_vals)), tpd_vals, color=BAR_COLOR, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax8.set_xlabel("Documents (ranked by table count)")
        ax8.set_ylabel("Number of tables")
        ax8.set_title("Tables per Document")
    else:
        ax8.text(0.5, 0.5, "No table data", ha="center", va="center", transform=ax8.transAxes)
        ax8.set_title("Tables per Document")

    # ── 9. Chunk text length — top 10 longest (outlier spotlight) ───────────
    ax9 = fig.add_subplot(gs[2, 2])
    text_key = sec_data.get("text_key", "section_text")
    sized_chunks = sorted(
        [(len(c.get(text_key, "")), c["document_id"]) for c in chunks],
        reverse=True
    )[:10]
    vals_9   = [v for v, _ in sized_chunks]
    labels_9 = [did[:8] for _, did in sized_chunks]
    colors_9 = [OUT_COLOR if v > high_size else BAR_COLOR for v in vals_9]
    ax9.barh(range(len(vals_9)), vals_9, color=colors_9, alpha=0.85, edgecolor="white")
    ax9.set_yticks(range(len(vals_9)))
    ax9.set_yticklabels(labels_9, fontsize=7, family="monospace")
    ax9.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax9.axvline(high_size, color=OUT_COLOR, linestyle="--", linewidth=1, alpha=0.7,
                label=f"IQR×1.5 ({high_size:,.0f})")
    ax9.legend(fontsize=7)
    ax9.set_xlabel("Chunk text length (chars)")
    ax9.set_title("Top 10 Longest Chunks\n(red = outlier)")
    ax9.invert_yaxis()

    # ── 10. Chunk length by section category (box plot) ─────────────────────
    ax10 = fig.add_subplot(gs[3, :])  # spans all 3 columns
    text_key = sec_data.get("text_key", "section_text")
    cat_sizes: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        cat = c.get("section_category", "other")
        cat_sizes[cat].append(len(c.get(text_key, "")))

    # Order by median descending so longest categories are on the left
    ordered_cats = sorted(cat_sizes, key=lambda c: sorted(cat_sizes[c])[len(cat_sizes[c])//2], reverse=True)
    box_data   = [cat_sizes[c] for c in ordered_cats]
    box_labels = [c.replace("_", "\n") for c in ordered_cats]

    bp = ax10.boxplot(
        box_data,
        vert=True,
        patch_artist=True,
        medianprops=dict(color="navy", linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, color=OUT_COLOR),
        whiskerprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BAR_COLOR)
        patch.set_alpha(0.7)

    ax10.set_xticks(range(1, len(ordered_cats) + 1))
    ax10.set_xticklabels(box_labels, fontsize=8)
    ax10.set_ylabel("Chunk text length (chars)")
    ax10.set_xlabel("Section category")
    ax10.set_title("Chunk Text Length Distribution by Section Category")
    ax10.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIG_COLOR)
    print(f"\n  {GREEN}Plot saved → {output_path}{RESET}")
    plt.close(fig)


# ── markdown report ──────────────────────────────────────────────────────────

def write_md_report(
    chunks: list[dict],
    sec_data: dict,
    ent_data: dict,
    res_data: dict,
    tbl_data: dict,
    png_path: Path,
    md_path: Path,
    total_issues: int,
) -> None:
    """Write a markdown report embedding the PNG and all summary tables."""
    from datetime import datetime, timezone

    png_rel = png_path.name  # same directory, so just the filename
    n_docs  = len(set(c["document_id"] for c in chunks))
    text_key = sec_data.get("text_key", "section_text")

    lines: list[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    # ── header ────────────────────────────────────────────────────────────────
    L(f"# Pipeline Validation Report")
    L(f"")
    L(f"**Source:** `{md_path.parent / md_path.stem.replace('_report', '')}.jsonl`  ")
    L(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    L(f"**Documents:** {n_docs} · **Chunks:** {len(chunks)} · "
      f"**Issues flagged:** {total_issues}")
    L()
    L(f"![Validation plots]({png_rel})")
    L()

    # ── task 1: sections ─────────────────────────────────────────────────────
    L("---")
    L("## Task 1 · Section Extraction & Categorization")
    L()

    st_cpd = _stats(sec_data["chunks_per_doc"])
    st_sz  = _stats(sec_data["chunk_sizes"])

    L("### Chunk counts")
    L()
    L("| Metric | Value |")
    L("|---|---|")
    L(f"| Documents | {n_docs} |")
    L(f"| Total chunks | {len(chunks):,} |")
    L(f"| Chunks/doc — min | {st_cpd['min']:.0f} |")
    L(f"| Chunks/doc — mean | {st_cpd['mean']:.1f} |")
    L(f"| Chunks/doc — median | {st_cpd['p50']:.0f} |")
    L(f"| Chunks/doc — p95 | {st_cpd['p95']:.0f} |")
    L(f"| Chunks/doc — max | {st_cpd['max']:.0f} |")
    L()

    L("### Category distribution")
    L()
    L("| Category | Chunks | Share |")
    L("|---|---:|---:|")
    cat_counts = sec_data["cat_counts"]
    total = len(chunks)
    all_cats = list(SECTION_CATEGORIES) + [k for k in cat_counts if k not in SECTION_CATEGORIES]
    for cat in all_cats:
        n = cat_counts.get(cat, 0)
        L(f"| `{cat}` | {n:,} | {100*n/total:.1f}% |")
    L()

    L("### Chunk text length — overall")
    L()
    L("| Stat | Chars |")
    L("|---|---:|")
    for k, label in [("min","Min"), ("mean","Mean"), ("p50","Median"), ("p95","p95"), ("max","Max")]:
        L(f"| {label} | {st_sz[k]:,.0f} |")
    L()

    L("### Chunk text length by section category")
    L()
    L("| Category | n | Min | Median | Mean | p95 | Max |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    text_key = sec_data.get("text_key", "section_text")
    cat_sizes: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        cat_sizes[c.get("section_category", "other")].append(len(c.get(text_key, "")))
    for cat in sorted(cat_sizes, key=lambda c: sorted(cat_sizes[c])[len(cat_sizes[c])//2], reverse=True):
        sv = _stats(cat_sizes[cat])
        L(f"| `{cat}` | {len(cat_sizes[cat])} | {sv['min']:,.0f} | "
          f"{sv['p50']:,.0f} | {sv['mean']:,.0f} | {sv['p95']:,.0f} | {sv['max']:,.0f} |")
    L()

    # chunk outliers
    high_cpd  = sec_data["high_cpd"]
    high_size = sec_data["high_size"]
    cpd_by_doc = sec_data["cpd_by_doc"]

    cpd_outliers = [(did, n) for did, n in cpd_by_doc.items() if n > high_cpd or n <= 2]
    if cpd_outliers:
        L("### Outlier documents — chunk count")
        L()
        L("| Document | Chunks | Flag |")
        L("|---|---:|---|")
        for did, n in sorted(cpd_outliers, key=lambda x: -x[1]):
            flag = "⬆ high" if n > high_cpd else "⬇ low"
            label = _doc_label(did, chunks)
            L(f"| `{label}` | {n} | {flag} |")
        L()

    size_outliers = sorted(
        {(c["document_id"], len(c.get(text_key, "")))
         for c in chunks if len(c.get(text_key, "")) > high_size},
        key=lambda x: -x[1]
    )[:10]
    if size_outliers:
        L("### Outlier chunks — very long text")
        L()
        L("| Document | Chars |")
        L("|---|---:|")
        for did, sz in size_outliers:
            L(f"| `{_doc_label(did, chunks)}` | {sz:,} |")
        L()

    # ── task 2: entities ─────────────────────────────────────────────────────
    L("---")
    L("## Task 2 · Entity Extraction")
    L()

    all_entities = [e for c in chunks for e in c.get("entities", [])]
    type_counts  = ent_data.get("type_counts", Counter())

    L("### Totals")
    L()
    L("| Entity type | Count | Share |")
    L("|---|---:|---:|")
    for label, cnt in type_counts.most_common():
        L(f"| `{label}` | {cnt:,} | {100*cnt/max(len(all_entities),1):.1f}% |")
    L()

    if ent_data.get("ent_per_chunk"):
        epc = ent_data["ent_per_chunk"]
        st_epc = _stats(epc)
        L("### Entities per chunk")
        L()
        L("| Stat | Value |")
        L("|---|---:|")
        for k, label in [("mean","Mean"), ("p50","Median"), ("p95","p95"), ("max","Max")]:
            L(f"| {label} | {st_epc[k]:.1f} |")
        L()

    # top entities per type
    for etype in ("company", "person", "monetary_value"):
        typed = [e["text"] for e in all_entities if e.get("entity_type") == etype]
        if not typed:
            continue
        L(f"### Top 10 `{etype}` entities")
        L()
        L("| Entity | Mentions |")
        L("|---|---:|")
        for name, cnt in Counter(typed).most_common(10):
            L(f"| {name} | {cnt:,} |")
        L()

    # entity outliers
    if ent_data.get("ent_per_chunk"):
        _, high_epc = _iqr_bounds([x for x in ent_data["ent_per_chunk"] if x > 0] or [0])
        noisy = [
            (c.get("document_id","?"), c.get("chunk_id", c.get("section_heading","?")), len(c["entities"]))
            for c in chunks if len(c.get("entities",[])) > high_epc
        ]
        if noisy:
            L(f"### Outlier chunks — high entity count (> {high_epc:.0f})")
            L()
            L("| Document | Chunk | Entities |")
            L("|---|---|---:|")
            for did, cid, n in sorted(noisy, key=lambda x: -x[2])[:10]:
                short_cid = str(cid)[:40]
                L(f"| `{_doc_label(did, chunks)}` | `{short_cid}` | {n:,} |")
            L()

    # ── task 3: resolution ────────────────────────────────────────────────────
    L("---")
    L("## Task 3 · Entity Resolution")
    L()

    raw_total = sum(len(c.get("entities", [])) for c in chunks)
    res_total = sum(len(c.get("resolved_entities", [])) for c in chunks)
    _RESOLVABLE = {"company", "person"}
    resolvable = sum(
        1 for c in chunks for e in c.get("entities", [])
        if e.get("entity_type", "").lower() in _RESOLVABLE
    )
    coverage = res_total / resolvable if resolvable else 0
    canonical_counts = res_data.get("canonical_counts", Counter())

    L("| Metric | Value | Definition |")
    L("|---|---:|---|")
    L(f"| Raw entities | {raw_total:,} | All extracted mentions (company + person + monetary_value) |")
    L(f"| Resolvable (company + person) | {resolvable:,} | Company + person mentions only — candidates for grouping |")
    L(f"| Resolved entries | {res_total:,} | Resolvable mentions assigned a canonical name |")
    L(f"| Coverage | {coverage:.1%} | Resolved / resolvable |")
    L(f"| Unique canonical names | {len(canonical_counts):,} | Distinct representative names after grouping variants |")
    L()

    if canonical_counts:
        singletons = sum(1 for v in canonical_counts.values() if v == 1)
        L(f"Singleton canonical names (appear in 1 chunk only): "
          f"**{singletons}** ({100*singletons/max(len(canonical_counts),1):.1f}%)")
        L()
        L("### Top 10 canonical entities")
        L()
        L("| Canonical name | Chunks |")
        L("|---|---:|")
        for name, cnt in canonical_counts.most_common(10):
            L(f"| {name} | {cnt:,} |")
        L()

    # ── resolution method breakdown ───────────────────────────────────────────
    method_counts = res_data.get("method_counts", Counter())
    method_defs = res_data.get("method_defs", {})
    if method_counts:
        L("### Resolution method breakdown")
        L()
        L("| Method | Count | % | Definition |")
        L("|---|---:|---:|---|")
        res_total_md = sum(method_counts.values())
        for method, cnt in method_counts.most_common():
            pct = 100 * cnt / res_total_md if res_total_md else 0
            defn = method_defs.get(method, "")
            L(f"| {method} | {cnt:,} | {pct:.1f}% | {defn} |")
        L()

    # ── resolution score distribution ─────────────────────────────────────────
    non_canonical_scores = res_data.get("non_canonical_scores", [])
    if non_canonical_scores:
        ss = _stats(non_canonical_scores)
        L("### Resolution score distribution *(excludes canonical/singleton)*")
        L()
        L(f"min **{ss['min']:.0f}** · p25 **{ss.get('p25',0):.0f}** · "
          f"p50 **{ss['p50']:.0f}** · p75 **{ss.get('p75',0):.0f}** · max **{ss['max']:.0f}**")
        L()
        L("- Score = 100 → exact / acronym match")
        L("- Score 90–99 → fuzzy match above threshold")
        L("- Score < 50 → cluster_link with weak direct evidence")
        L()

    # ── per-method examples ────────────────────────────────────────────────────
    examples_by_method = res_data.get("examples_by_method", {})
    if examples_by_method:
        L("### One example per resolution method")
        L()
        L("| Method | Matched text | Canonical name | Score |")
        L("|---|---|---|---:|")
        for method, e in examples_by_method.items():
            matched = (e.get("matched_text", "") or "")[:50].replace("|", "\\|").replace("\n", " ")
            canonical = (e.get("canonical_name", "") or "")[:50].replace("|", "\\|").replace("\n", " ")
            score = e.get("resolution_score", "–")
            L(f"| {method} | {matched} | {canonical} | {score} |")
        L()

    # ── task 4: tables ────────────────────────────────────────────────────────
    L("---")
    L("## Task 4 · Table Detection & Parsing")
    L()

    all_tables = [t for c in chunks for t in c.get("tables", [])]
    chunks_with_tbl = sum(1 for c in chunks if c.get("tables"))
    by_doc_tbl: Counter = Counter()
    for c in chunks:
        by_doc_tbl[c["document_id"]] += len(c.get("tables", []))
    docs_with_tbl = sum(1 for v in by_doc_tbl.values() if v > 0)

    # Core metrics
    L("| Metric | Value |")
    L("|---|---|")
    L(f"| Total tables | {len(all_tables):,} |")
    L(f"| Chunks with ≥1 table | {chunks_with_tbl:,} ({100*chunks_with_tbl/max(len(chunks),1):.1f}%) |")
    L(f"| Documents with ≥1 table | {docs_with_tbl}/{n_docs} |")
    if all_tables:
        with_cols_md = sum(1 for t in all_tables if t.get("columns"))
        structured_md = sum(1 for t in all_tables if t.get("rows") and isinstance(t["rows"][0], dict))
        with_sum = sum(1 for t in all_tables if t.get("summary", "").strip())
        llm_sum  = sum(1 for t in all_tables if t.get("summary_used_llm"))
        rs = _stats([len(t.get("rows", [])) for t in all_tables])
        cs = _stats([len(t.get("columns", [])) for t in all_tables])
        L(f"| Column detection rate | {with_cols_md:,}/{len(all_tables):,} ({100*with_cols_md/len(all_tables):.1f}%) |")
        L(f"| Rows structured as dicts | {structured_md:,}/{len(all_tables):,} ({100*structured_md/len(all_tables):.1f}%) |")
        L(f"| Tables with summary | {with_sum:,} ({100*with_sum/len(all_tables):.1f}%) |")
        L(f"| LLM summaries | {llm_sum:,} ({100*llm_sum/len(all_tables):.1f}%) |")
        L(f"| Rows/table — mean / p50 / max | {rs['mean']:.1f} / {rs['p50']:.0f} / {rs['max']:.0f} |")
        L(f"| Cols/table — mean / p50 / max | {cs['mean']:.1f} / {cs['p50']:.0f} / {cs['max']:.0f} |")
    L()

    # Table type breakdown
    if all_tables:
        type_counts_md: Counter = Counter(t.get("table_type", "unknown") for t in all_tables)
        L("### Table type breakdown")
        L()
        L("| Type | Count | % |")
        L("|---|---:|---:|")
        for ttype, cnt in type_counts_md.most_common():
            L(f"| {ttype} | {cnt:,} | {100*cnt/len(all_tables):.1f}% |")
        L()

    # By section category
    cat_tables_md: dict[str, list] = {}
    for c in chunks:
        cat = c.get("section_category", "other")
        cat_tables_md.setdefault(cat, []).extend(c.get("tables", []))

    if cat_tables_md:
        L("### Tables by section category")
        L()
        L("| Category | Tables | Sections w/ table | Avg rows | Col detect |")
        L("|---|---:|---:|---:|---:|")
        for cat in sorted(cat_tables_md, key=lambda c: -len(cat_tables_md[c])):
            tbls = cat_tables_md[cat]
            cat_chunks = [c for c in chunks if c.get("section_category") == cat]
            cat_with = sum(1 for c in cat_chunks if c.get("tables"))
            avg_rows = sum(len(t.get("rows", [])) for t in tbls) / len(tbls) if tbls else 0
            col_rate = sum(1 for t in tbls if t.get("columns")) / len(tbls) if tbls else 0
            L(f"| {cat} | {len(tbls):,} | {_pct(cat_with, len(cat_chunks))} | {avg_rows:.1f} | {col_rate:.0%} |")
        L()

    # LLM cost
    cost_stats = tbl_data.get("cost_stats", {})
    if cost_stats:
        L("### LLM cost summary")
        L()
        if cost_stats.get("run_total_logged"):
            hit_pct = 100 * cost_stats["run_hit_rate"]
            L("**Last run**")
            L()
            L("| Metric | Value |")
            L("|---|---:|")
            L(f"| Fresh API calls | {cost_stats['run_fresh_calls']:,} |")
            L(f"| Cache hits | {cost_stats['run_cache_hits']:,} ({hit_pct:.1f}%) |")
            if cost_stats["run_errors"]:
                L(f"| Errors | {cost_stats['run_errors']:,} |")
            L()
        L("**Cumulative (all runs)**")
        L()
        L("| Metric | Value |")
        L("|---|---:|")
        L(f"| Unique API calls | {cost_stats['total_entries']:,} |")
        L(f"| — named-column summaries | {cost_stats['summary_entries']:,} |")
        L(f"| — generic-column inference | {cost_stats['infer_entries']:,} |")
        L(f"| Prompt tokens | {cost_stats['prompt_tokens']:,} |")
        L(f"| Completion tokens | {cost_stats['completion_tokens']:,} |")
        L(f"| Total tokens | {cost_stats['total_tokens']:,} |")
        L(f"| Estimated cost | ${cost_stats['estimated_cost_usd']:.4f} |")
        L()

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {GREEN}Markdown saved → {md_path}{RESET}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pipeline JSONL output at any stage.")
    parser.add_argument(
        "--input", type=Path, default=OUTPUT_JSONL,
        help=f"JSONL file to validate (default: {OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating the PNG report (useful in CI or headless environments)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run scripts/run_pipeline.py first.")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"\n{BOLD}Loaded {len(chunks)} chunks from {args.input}{RESET}")

    total_issues = 0

    issues_s, sec_data = validate_sections(chunks)
    total_issues += issues_s

    issues_e, ent_data = validate_entities(chunks)
    total_issues += issues_e

    issues_r, res_data = validate_entity_resolution(chunks)
    total_issues += issues_r

    issues_t, tbl_data = validate_tables(chunks)
    total_issues += issues_t

    _h("SUMMARY")
    if total_issues == 0:
        _ok("All checks passed — no issues found")
    else:
        _warn(f"{total_issues} issue(s) flagged — review warnings above")
    print()

    eval_dir = Path(__file__).parent.parent / "outputs" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    plot_path = eval_dir / f"{stem}_report.png"
    md_path   = eval_dir / f"{stem}_report.md"

    if not args.no_plot:
        try:
            build_report(chunks, sec_data, ent_data, tbl_data, plot_path)
        except ImportError:
            _warn("matplotlib not installed — skipping plot (pip install matplotlib)")

    write_md_report(
        chunks, sec_data, ent_data, res_data, tbl_data,
        plot_path, md_path, total_issues,
    )


if __name__ == "__main__":
    main()
