"""Table detection and parsing for SEC 10-K text files.

SEC EDGAR text-extraction quirks this parser handles:
- Non-breaking spaces (\xa0) used as column separators
- Multi-line rows: label on one line, each value on its own continuation line
- Column headers appear 1-8 lines before OR after the "(In millions)" anchor
- Footnote references like "(a)", "(1)" interspersed between data rows
- Page numbers and "Table of Contents" lines mid-table

Approach:
1. Normalize \xa0 → space throughout.
2. Detect candidate regions via "(In millions/thousands)" anchors.
3. Search ±8 lines around the anchor for a year/period column header.
4. Parse rows using a state machine: labeled lines start new rows; value-only
   continuation lines are appended to the current row's values.
5. Align collected values to columns when count matches; otherwise keep as list.
6. Produce structured JSON with table_type, units, columns, rows, summary.
7. Optional OpenAI summary via cached LLM call.
"""

import logging
import csv
import hashlib
import json
import re
import time
from typing import Optional

from src.constants import CACHE_DIR, COSTS_DIR, OPENAI_MODEL

logger = logging.getLogger(__name__)

# ── detection patterns ────────────────────────────────────────────────────────

_IN_UNITS_RE = re.compile(
    r"\((?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)[^)]*\)",
    re.IGNORECASE,
)

# Column header: line containing 2+ year references or period labels
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Pipe-separated table row
_PIPE_ROW_RE = re.compile(r"\|[^|\n]+\|")

# Value-only line: starts with optional whitespace/$ then a number or "(number)"
_VALUE_ONLY_RE = re.compile(r"^[\s\xa0$%().,\-—–]*[\d][\d\s$%().,\-—–]*$")

# Footnote line: "(a) ...", "(1) ..." — skip these
_FOOTNOTE_RE = re.compile(r"^\s*\([a-zA-Z1-9]\)\s")

# Page number or boilerplate line to skip
_SKIP_LINE_RE = re.compile(
    r"^\s*(?:\d{1,4}\s*$"                  # bare page number
    r"|table\s+of\s+contents"              # TOC marker
    r"|[-─═]{10,}"                         # separator line
    r")",
    re.IGNORECASE,
)

_TABLE_LLM_SUMMARY_CACHE_JSON = CACHE_DIR / "table_llm_summary_cache.json"
_TABLE_LLM_SUMMARY_METRICS_CSV = COSTS_DIR / "table_llm_summary_metrics.csv"
_TABLE_LLM_SUMMARY_CACHE_VERSION = "table_summary_v2"

_GPT41_MINI_INPUT_COST_PER_1M_TOKENS = 0.40
_GPT41_MINI_OUTPUT_COST_PER_1M_TOKENS = 1.60


# ── public interface ──────────────────────────────────────────────────────────

def detect_and_parse_tables(chunk: dict, use_llm_summaries: bool = False) -> list[dict]:
    """Detect tables in section_text and return a list of structured table dicts."""
    raw_text = chunk.get("section_text", "")
    text = _normalize(raw_text)
    tables = []

    # Strategy 1: anchor on "(In millions/thousands)" markers
    seen_starts: set[int] = set()
    for m in _IN_UNITS_RE.finditer(text):
        anchor = m.start()
        # Look back up to 800 chars and forward up to 4000 chars for context
        region_start = max(0, anchor - 800)
        region_end = min(len(text), anchor + 4000)
        candidate = text[region_start:region_end]
        parsed = _parse_space_table(candidate, units_hint=m.group(0))
        if parsed:
            key = anchor // 200  # deduplicate overlapping windows
            if key not in seen_starts:
                seen_starts.add(key)
                _apply_summary(parsed, chunk, use_llm_summaries)
                tables.append(parsed)

    # Strategy 2: pipe-separated tables
    for block in _find_pipe_blocks(text):
        parsed = _parse_pipe_table(block)
        if parsed:
            _apply_summary(parsed, chunk, use_llm_summaries)
            tables.append(parsed)

    # Deduplicate by first 120 chars of raw_text
    seen_raw: set[str] = set()
    unique: list[dict] = []
    for t in tables:
        key = t.get("raw_text", "")[:120]
        if key not in seen_raw:
            seen_raw.add(key)
            unique.append(t)

    return unique


# ── text normalization ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Replace non-breaking spaces and normalize unicode dashes."""
    return (
        text
        .replace("\xa0", " ")
        .replace("—", "--")
        .replace("–", "-")
        .replace("−", "-")
    )


# ── space-aligned financial tables ───────────────────────────────────────────

def _parse_space_table(text: str, units_hint: str = "") -> Optional[dict]:
    lines = text.splitlines()
    if not lines:
        return None

    # 1. Locate the units anchor line
    anchor_idx = next(
        (i for i, l in enumerate(lines) if _IN_UNITS_RE.search(l)),
        0,
    )

    # 2. Search ±8 lines for a column header with year or period labels.
    #    Also check the anchor line itself (years may appear on the same line as units).
    columns: list[str] = []
    col_header_idx: Optional[int] = None
    search_lo = max(0, anchor_idx - 8)
    search_hi = min(len(lines), anchor_idx + 9)

    # Strip the units token from the anchor line before checking for year cols
    anchor_line_stripped = _IN_UNITS_RE.sub("", lines[anchor_idx]).strip()
    if anchor_line_stripped:
        candidate_cols = _try_parse_column_header(anchor_line_stripped)
        if candidate_cols and len(candidate_cols) >= 2:
            col_header_idx = anchor_idx
            columns = candidate_cols

    for i in range(search_lo, search_hi):
        if i == anchor_idx:
            continue
        candidate_cols = _try_parse_column_header(lines[i])
        if candidate_cols and len(candidate_cols) >= 2:
            # Prefer headers closer to the anchor
            if col_header_idx is None or abs(i - anchor_idx) < abs(col_header_idx - anchor_idx):
                col_header_idx = i
                columns = candidate_cols

    # 3. Data rows start after the later of anchor/col_header
    data_start = max(anchor_idx, col_header_idx if col_header_idx is not None else anchor_idx) + 1

    # 4. State-machine row builder
    rows, raw_lines = _collect_rows(lines, data_start)

    if len(rows) < 2:
        return None

    # 5. Align rows to columns
    if columns:
        aligned = _align_rows(rows, columns)
    else:
        # Infer column count from the most common row value count
        columns = _infer_columns(rows)
        aligned = _align_rows(rows, columns) if columns else [r for r in rows]

    raw_text = "\n".join(lines[max(0, anchor_idx - 2): anchor_idx + len(raw_lines) + 4])
    summary = _deterministic_summary(columns, aligned, units_hint)

    return {
        "table_type": "financial",
        "units": units_hint,
        "columns": columns,
        "rows": aligned,
        "raw_text": raw_text,
        "summary": summary,
    }


def _try_parse_column_header(line: str) -> list[str]:
    """Return column labels if this line looks like a table column header."""
    stripped = line.strip()
    if not stripped:
        return []
    # Must contain at least one year or a recognizable period keyword
    has_year = bool(_YEAR_RE.search(stripped))
    has_period = bool(re.search(
        r"\b(?:year|quarter|q[1-4]|ended|ending|ytd|fiscal|period|months?)\b",
        stripped, re.IGNORECASE,
    ))
    if not (has_year or has_period):
        return []
    parts = re.split(r"\s{2,}|\t", stripped)
    return [p.strip() for p in parts if p.strip()]


def _is_value_only(line: str) -> bool:
    """True if a line contains only numeric/currency content (no meaningful label)."""
    stripped = line.strip()
    if not stripped:
        return False
    if _FOOTNOTE_RE.match(stripped):
        return False
    return bool(_VALUE_ONLY_RE.match(stripped))


def _extract_cells(line: str) -> list[str]:
    """Extract all numeric cell values from a line."""
    return re.findall(r"\(?\$?\s*[\d,]+(?:\.\d+)?\s*%?\)?", line)


def _extract_label(line: str) -> str:
    m = re.match(r"^([^$\d(]+?)(?=\s*[\$\d(]|$)", line.strip())
    return m.group(1).strip() if m else line.strip()


def _collect_rows(lines: list[str], start: int) -> tuple[list, list[str]]:
    """
    State-machine parser: labeled lines start a row; value-only continuation
    lines extend the current row's values. Returns (rows, raw_lines_used).
    """
    current_label: Optional[str] = None
    current_values: list[str] = []
    rows: list = []
    raw_lines: list[str] = []

    def _emit():
        if current_label is not None and current_values:
            rows.append({"_label": current_label, "_values": current_values[:]})

    for line in lines[start:]:
        stripped = line.strip()

        # Stop on a new section heading
        if re.match(r"^(?:ITEM|PART)\s+\d", stripped, re.IGNORECASE):
            break
        # Skip boilerplate / page markers
        if _SKIP_LINE_RE.match(stripped):
            continue
        # Skip blank lines between rows (don't reset state)
        if not stripped:
            continue
        # Skip footnote lines
        if _FOOTNOTE_RE.match(stripped):
            continue

        if _is_value_only(stripped):
            # Continuation values for the current row
            current_values.extend(_extract_cells(stripped))
            raw_lines.append(line)
        else:
            # New labeled row
            _emit()
            current_label = _extract_label(stripped)
            current_values = _extract_cells(stripped)
            raw_lines.append(line)

    _emit()
    return rows, raw_lines


def _infer_columns(rows: list) -> list[str]:
    """Infer column count from most-common value-list length; return generic headers."""
    from collections import Counter
    lengths = Counter(len(r["_values"]) for r in rows if r["_values"])
    if not lengths:
        return []
    most_common_len = lengths.most_common(1)[0][0]
    if most_common_len < 1:
        return []
    # Generic headers: Item + Year1, Year2, ...
    return ["Item"] + [f"Col{i+1}" for i in range(most_common_len)]


def _align_rows(rows: list, columns: list[str]) -> list[dict]:
    """
    Align each row's values to the column list.

    Convention:
    - columns[0] is treated as the label/item column.
    - The remaining N-1 columns receive the row's values.
    - If a row has exactly len(columns) values (no separate label), map 1:1.
    - Otherwise pad / truncate to fit.
    """
    aligned = []
    n_value_cols = len(columns) - 1  # number of value columns

    for row in rows:
        label = row.get("_label", "")
        values = row.get("_values", [])

        if not label and not values:
            continue

        if len(values) == len(columns):
            # All values including label column present
            aligned.append(dict(zip(columns, values)))
        elif n_value_cols > 0:
            record: dict = {columns[0]: label}
            for i, col in enumerate(columns[1:]):
                record[col] = values[i] if i < len(values) else ""
            aligned.append(record)
        else:
            aligned.append({columns[0]: label})

    return aligned


def _deterministic_summary(columns: list[str], rows: list, units: str) -> str:
    n_rows = len(rows)
    n_cols = len(columns)
    units_str = f" ({units})" if units else ""
    value_cols = [c for c in columns if c != "Item" and not c.startswith("Col")]
    if value_cols:
        col_desc = ", ".join(value_cols[:4])
        extra = f" and {n_cols - 5} more" if n_cols > 5 else ""
        return (
            f"Table with {n_rows} row{'s' if n_rows != 1 else ''}{units_str} "
            f"across periods: {col_desc}{extra}."
        )
    if columns:
        return (
            f"Table with {n_rows} row{'s' if n_rows != 1 else ''}{units_str} "
            f"and {n_cols} column{'s' if n_cols != 1 else ''}."
        )
    return f"Table with {n_rows} row{'s' if n_rows != 1 else ''}{units_str}."


# ── pipe-separated tables ─────────────────────────────────────────────────────

def _find_pipe_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks, current = [], []
    for line in lines:
        if _PIPE_ROW_RE.search(line):
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append("\n".join(current))
            current = []
    if len(current) >= 2:
        blocks.append("\n".join(current))
    return blocks


def _parse_pipe_table(block: str) -> Optional[dict]:
    lines = [l for l in block.splitlines() if _PIPE_ROW_RE.search(l)]
    if len(lines) < 2:
        return None

    def split_pipe(line: str) -> list[str]:
        parts = line.split("|")
        return [p.strip() for p in parts if p.strip()]

    header_cells = split_pipe(lines[0])
    data_rows: list[dict] = []
    for line in lines[1:]:
        if re.match(r"^[\|\-\s]+$", line):
            continue
        cells = split_pipe(line)
        if cells:
            if len(cells) == len(header_cells):
                data_rows.append(dict(zip(header_cells, cells)))
            else:
                data_rows.append(dict(enumerate(cells)))

    if not data_rows:
        return None

    summary = _deterministic_summary(header_cells, data_rows, "")
    return {
        "table_type": "pipe",
        "units": "",
        "columns": header_cells,
        "rows": data_rows,
        "raw_text": block,
        "summary": summary,
    }


# ── LLM summary (optional) ────────────────────────────────────────────────────

def _has_generic_columns(columns: list[str]) -> bool:
    """True when all value columns are generic placeholders (Col1, Col2, ...)."""
    value_cols = [c for c in columns if c not in ("Item",)]
    return bool(value_cols) and all(re.match(r"^Col\d+$", c) for c in value_cols)


def _apply_summary(parsed: dict, chunk: dict, use_llm_summaries: bool) -> None:
    """Attach summary provenance fields; optionally replace summary via LLM.

    Two LLM paths:
    - Named columns → ask for a summary only.
    - Generic columns (Col1/Col2) → ask for column inference + summary from raw_text.
      If the LLM recovers column names, remap the row dicts in place.
    """
    parsed["summary_used_llm"] = False
    parsed["summary_source"] = "deterministic"
    parsed["summary_cache_status"] = ""

    if not use_llm_summaries or not parsed.get("columns"):
        return

    heading = chunk.get("section_heading", "")

    if _has_generic_columns(parsed["columns"]):
        # Send raw_text so the LLM can read the original column headers
        result = _llm_infer_columns_cached(parsed, heading)
        if result["used_llm"]:
            if result.get("columns"):
                _remap_columns(parsed, result["columns"])
            parsed["summary"] = result["summary"]
            parsed["summary_used_llm"] = True
            parsed["summary_source"] = "llm_inferred"
            parsed["summary_cache_status"] = result["cache_status"]
    else:
        result = _llm_table_summary_cached(parsed, heading)
        if result["used_llm"]:
            parsed["summary"] = result["summary"]
            parsed["summary_used_llm"] = True
            parsed["summary_source"] = "llm"
            parsed["summary_cache_status"] = result["cache_status"]


def _remap_columns(parsed: dict, new_columns: list[str]) -> None:
    """Replace generic column names in parsed['columns'] and all row dicts."""
    old_columns = parsed["columns"]
    if len(new_columns) != len(old_columns):
        return
    mapping = dict(zip(old_columns, new_columns))
    parsed["columns"] = new_columns
    parsed["rows"] = [
        {mapping.get(k, k): v for k, v in row.items()}
        if isinstance(row, dict) else row
        for row in parsed.get("rows", [])
    ]


def _llm_table_summary(parsed_table: dict, section_heading: str) -> str:
    """Backward-compatible wrapper returning only the summary text."""
    return _llm_table_summary_cached(parsed_table, section_heading)["summary"]


# ── column inference via LLM (generic-column tables) ─────────────────────────

def _infer_payload(parsed_table: dict, section_heading: str) -> dict:
    return {
        "section_heading": section_heading,
        "units": parsed_table.get("units", ""),
        # Only the raw table snippet — no full section text
        "raw_text": parsed_table.get("raw_text", "")[:600],
        "n_columns": len(parsed_table.get("columns", [])),
    }


def _infer_cache_key(parsed_table: dict, section_heading: str, model: str) -> str:
    payload = _infer_payload(parsed_table, section_heading)
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    raw = json.dumps({
        "cache_version": _TABLE_LLM_SUMMARY_CACHE_VERSION + "_infer",
        "model": model,
        "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _llm_infer_columns_cached(parsed_table: dict, section_heading: str) -> dict:
    """Return LLM-inferred column names + summary, using content-addressed cache."""
    fallback_summary = parsed_table.get("summary", "")
    cache_key = _infer_cache_key(parsed_table, section_heading, OPENAI_MODEL)
    cache = _load_table_summary_cache()

    if cache_key in cache:
        entry = cache[cache_key]
        if entry.get("summary"):
            _append_table_cache_hit_metrics(entry)
            return {
                "summary": entry["summary"],
                "columns": entry.get("columns", []),
                "used_llm": True,
                "cache_status": "hit",
            }

    result = _llm_infer_columns_uncached(parsed_table, section_heading)
    if not result.get("summary"):
        return {"summary": fallback_summary, "columns": [], "used_llm": False, "cache_status": "miss"}

    payload_json = json.dumps(_infer_payload(parsed_table, section_heading), ensure_ascii=False, sort_keys=True)
    cache[cache_key] = {
        "cache_version": _TABLE_LLM_SUMMARY_CACHE_VERSION + "_infer",
        "model": result.get("model", OPENAI_MODEL),
        "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        "prompt_chars": result.get("prompt_chars", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "estimated_cost_usd": result.get("estimated_cost_usd", 0.0),
        "summary": result["summary"],
        "columns": result.get("columns", []),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_table_summary_cache(cache)
    return {
        "summary": result["summary"],
        "columns": result.get("columns", []),
        "used_llm": True,
        "cache_status": "miss",
    }


def _llm_infer_columns_uncached(parsed_table: dict, section_heading: str) -> dict:
    """Ask the LLM to infer column names from the raw table snippet and summarize."""
    started_at = time.perf_counter()
    try:
        from openai import OpenAI
        from src.constants import OPENAI_API_KEY, OPENAI_MODEL

        if not OPENAI_API_KEY:
            return {"summary": ""}

        client = OpenAI(api_key=OPENAI_API_KEY)
        units = parsed_table.get("units", "")
        units_note = f" Values are {units}." if units else ""
        n_cols = len(parsed_table.get("columns", []))
        # Send only the raw table snippet — already bounded to 600 chars in the payload
        raw_snippet = parsed_table.get("raw_text", "")[:600]

        prompt = (
            f"You are a financial analyst reading a SEC 10-K table.\n"
            f"Section: {section_heading!r}.{units_note}\n\n"
            f"Raw table text (excerpt):\n{raw_snippet}\n\n"
            f"The table has {n_cols} columns. The automated parser could not identify "
            f"the column headers from this text.\n\n"
            f"Reply with JSON only, no explanation:\n"
            f'{{"columns": ["<col1>", "<col2>", ...], "summary": "<1-2 sentences>"}}\n\n'
            f"Rules:\n"
            f"- Return exactly {n_cols} column names inferred from the raw text above.\n"
            f"- If you cannot determine a column name, use null for that entry.\n"
            f"- summary: name the metric(s), time periods, and any notable trend. "
            f"Preserve numeric values exactly. Do not invent information."
        )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw_response = resp.choices[0].message.content.strip()
        parsed_resp = json.loads(raw_response)
        summary = parsed_resp.get("summary", "")
        inferred_cols = parsed_resp.get("columns", [])
        # Filter out nulls; keep list length matching n_cols
        if isinstance(inferred_cols, list) and len(inferred_cols) == n_cols:
            inferred_cols = [c if c and c != "null" else f"Col{i+1}"
                             for i, c in enumerate(inferred_cols)]
        else:
            inferred_cols = []

        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or (prompt_tokens + completion_tokens)
        )
        estimated_cost = _estimate_table_llm_cost(prompt_tokens, completion_tokens)
        _append_table_llm_metrics({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": getattr(resp, "model", OPENAI_MODEL),
            "status": "ok_infer",
            "prompt_chars": len(prompt),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "response_chars": len(raw_response),
            "estimated_cost_usd": round(estimated_cost, 8),
            "error": "",
        })
        return {
            "summary": summary,
            "columns": inferred_cols,
            "model": getattr(resp, "model", OPENAI_MODEL),
            "prompt_chars": len(prompt),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
        }
    except Exception as e:
        _append_table_llm_metrics({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": OPENAI_MODEL,
            "status": "error_infer",
            "prompt_chars": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "response_chars": 0, "estimated_cost_usd": 0.0,
            "error": str(e),
        })
        logger.warning("LLM column inference failed: %s", e)
        return {"summary": ""}


def _estimate_table_llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1_000_000) * _GPT41_MINI_INPUT_COST_PER_1M_TOKENS
        + (completion_tokens / 1_000_000) * _GPT41_MINI_OUTPUT_COST_PER_1M_TOKENS
    )


def _table_summary_payload(parsed_table: dict, section_heading: str) -> dict:
    return {
        "section_heading": section_heading,
        "units": parsed_table.get("units", ""),
        "columns": parsed_table.get("columns", []),
        "rows": parsed_table.get("rows", [])[:15],
    }


def _table_summary_cache_key(parsed_table: dict, section_heading: str, model: str) -> str:
    payload = _table_summary_payload(parsed_table, section_heading)
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    raw = json.dumps({
        "cache_version": _TABLE_LLM_SUMMARY_CACHE_VERSION,
        "model": model,
        "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_table_summary_cache() -> dict:
    if not _TABLE_LLM_SUMMARY_CACHE_JSON.exists():
        return {}
    try:
        return json.loads(_TABLE_LLM_SUMMARY_CACHE_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Ignoring unreadable table summary cache: %s", e)
        return {}


def _write_table_summary_cache(cache: dict) -> None:
    _TABLE_LLM_SUMMARY_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = _TABLE_LLM_SUMMARY_CACHE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(_TABLE_LLM_SUMMARY_CACHE_JSON)  # atomic rename on same filesystem


def _append_table_llm_metrics(row: dict) -> None:
    _TABLE_LLM_SUMMARY_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = (
        _TABLE_LLM_SUMMARY_METRICS_CSV.exists()
        and _TABLE_LLM_SUMMARY_METRICS_CSV.stat().st_size > 0
    )
    fieldnames = [
        "timestamp_utc", "model", "status", "prompt_chars",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "latency_ms", "response_chars", "estimated_cost_usd", "error",
    ]
    with _TABLE_LLM_SUMMARY_METRICS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_table_cache_hit_metrics(entry: dict) -> None:
    _append_table_llm_metrics({
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": entry.get("model", OPENAI_MODEL),
        "status": "cache_hit",
        "prompt_chars": entry.get("prompt_chars", 0),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0,
        "response_chars": len(entry.get("summary", "")),
        "estimated_cost_usd": 0.0,
        "error": "",
    })


def _llm_table_summary_cached(parsed_table: dict, section_heading: str) -> dict:
    """Return an LLM summary using a content-addressed cache when possible."""
    fallback = parsed_table.get("summary", "")
    cache_key = _table_summary_cache_key(parsed_table, section_heading, OPENAI_MODEL)
    cache = _load_table_summary_cache()

    if cache_key in cache:
        entry = cache[cache_key]
        summary = entry.get("summary", "")
        if summary:
            _append_table_cache_hit_metrics(entry)
            return {"summary": summary, "used_llm": True, "cache_status": "hit"}

    result = _llm_table_summary_uncached(parsed_table, section_heading)
    summary = result.get("summary", "")
    if not summary:
        return {"summary": fallback, "used_llm": False, "cache_status": "miss"}

    payload_json = json.dumps(
        _table_summary_payload(parsed_table, section_heading),
        ensure_ascii=False, sort_keys=True,
    )
    cache[cache_key] = {
        "cache_version": _TABLE_LLM_SUMMARY_CACHE_VERSION,
        "model": result.get("model", OPENAI_MODEL),
        "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        "prompt_chars": result.get("prompt_chars", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "estimated_cost_usd": result.get("estimated_cost_usd", 0.0),
        "summary": summary,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_table_summary_cache(cache)
    return {"summary": summary, "used_llm": True, "cache_status": "miss"}


def _llm_table_summary_uncached(parsed_table: dict, section_heading: str) -> dict:
    """Call OpenAI to produce a 1–2 sentence table summary."""
    started_at = time.perf_counter()
    try:
        from openai import OpenAI
        from src.constants import OPENAI_API_KEY, OPENAI_MODEL

        if not OPENAI_API_KEY:
            return {"summary": ""}

        client = OpenAI(api_key=OPENAI_API_KEY)
        units = parsed_table.get("units", "")
        units_note = f" Values are {units}." if units else ""
        table_json = json.dumps(
            {
                "columns": parsed_table.get("columns", []),
                "rows": parsed_table.get("rows", [])[:15],
            },
            ensure_ascii=False,
        )
        prompt = (
            f"You are a financial analyst summarizing a table from a SEC 10-K filing.\n"
            f"Section: {section_heading!r}.{units_note}\n\n"
            f"Table (JSON):\n{table_json}\n\n"
            "Write 1–2 sentences describing what this table shows. "
            "Name the metric(s), the time periods covered, and any notable trend. "
            "Do not infer information not present. Preserve numeric values exactly."
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
        )
        summary = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or (prompt_tokens + completion_tokens)
        )
        estimated_cost = _estimate_table_llm_cost(prompt_tokens, completion_tokens)
        _append_table_llm_metrics({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": getattr(resp, "model", OPENAI_MODEL),
            "status": "ok",
            "prompt_chars": len(prompt),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "response_chars": len(summary),
            "estimated_cost_usd": round(estimated_cost, 8),
            "error": "",
        })
        return {
            "summary": summary,
            "model": getattr(resp, "model", OPENAI_MODEL),
            "prompt_chars": len(prompt),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
        }
    except Exception as e:
        _append_table_llm_metrics({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": OPENAI_MODEL,
            "status": "error",
            "prompt_chars": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "response_chars": 0, "estimated_cost_usd": 0.0,
            "error": str(e),
        })
        logger.warning("LLM table summary failed: %s", e)
        return {"summary": ""}


# ── async batch processing ────────────────────────────────────────────────────

import asyncio

# Lock serialises cache writes so concurrent coroutines don't clobber each other.
# Reads are unlocked — reading a slightly stale cache is harmless (worst case: one
# redundant API call whose result overwrites the cache with the same value).
_cache_write_lock = asyncio.Lock()


async def _llm_table_summary_async(
    parsed_table: dict,
    section_heading: str,
    client,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Async version of _llm_table_summary_cached."""
    fallback = parsed_table.get("summary", "")
    cache_key = _table_summary_cache_key(parsed_table, section_heading, OPENAI_MODEL)

    # Cache read is sync and fast — no lock needed
    cache = _load_table_summary_cache()
    if cache_key in cache:
        entry = cache[cache_key]
        if entry.get("summary"):
            _append_table_cache_hit_metrics(entry)
            return {"summary": entry["summary"], "used_llm": True, "cache_status": "hit"}

    async with semaphore:
        started_at = time.perf_counter()
        try:
            units = parsed_table.get("units", "")
            units_note = f" Values are {units}." if units else ""
            table_json = json.dumps(
                {"columns": parsed_table.get("columns", []), "rows": parsed_table.get("rows", [])[:15]},
                ensure_ascii=False,
            )
            prompt = (
                f"You are a financial analyst summarizing a table from a SEC 10-K filing.\n"
                f"Section: {section_heading!r}.{units_note}\n\n"
                f"Table (JSON):\n{table_json}\n\n"
                "Write 1–2 sentences describing what this table shows. "
                "Name the metric(s), the time periods covered, and any notable trend. "
                "Do not infer information not present. Preserve numeric values exactly."
            )
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0,
            )
            summary = resp.choices[0].message.content.strip()
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
            estimated_cost = _estimate_table_llm_cost(prompt_tokens, completion_tokens)
            _append_table_llm_metrics({
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": getattr(resp, "model", OPENAI_MODEL),
                "status": "ok",
                "prompt_chars": len(prompt),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "response_chars": len(summary),
                "estimated_cost_usd": round(estimated_cost, 8),
                "error": "",
            })
            if not summary:
                return {"summary": fallback, "used_llm": False, "cache_status": "miss"}

            payload_json = json.dumps(_table_summary_payload(parsed_table, section_heading), ensure_ascii=False, sort_keys=True)
            entry = {
                "cache_version": _TABLE_LLM_SUMMARY_CACHE_VERSION,
                "model": getattr(resp, "model", OPENAI_MODEL),
                "payload_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
                "prompt_chars": len(prompt),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(estimated_cost, 8),
                "summary": summary,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            async with _cache_write_lock:
                cache = _load_table_summary_cache()
                cache[cache_key] = entry
                _write_table_summary_cache(cache)
            return {"summary": summary, "used_llm": True, "cache_status": "miss"}

        except Exception as e:
            _append_table_llm_metrics({
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": OPENAI_MODEL, "status": "error",
                "prompt_chars": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "response_chars": 0, "estimated_cost_usd": 0.0, "error": str(e),
            })
            logger.warning("Async LLM summary failed: %s", e)
            return {"summary": fallback, "used_llm": False, "cache_status": "miss"}


async def _llm_infer_columns_async(
    parsed_table: dict,
    section_heading: str,
    client,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Async version of _llm_infer_columns_cached."""
    fallback_summary = parsed_table.get("summary", "")
    cache_key = _infer_cache_key(parsed_table, section_heading, OPENAI_MODEL)

    cache = _load_table_summary_cache()
    if cache_key in cache:
        entry = cache[cache_key]
        if entry.get("summary"):
            _append_table_cache_hit_metrics(entry)
            return {"summary": entry["summary"], "columns": entry.get("columns", []), "used_llm": True, "cache_status": "hit"}

    async with semaphore:
        started_at = time.perf_counter()
        try:
            units = parsed_table.get("units", "")
            units_note = f" Values are {units}." if units else ""
            n_cols = len(parsed_table.get("columns", []))
            raw_snippet = parsed_table.get("raw_text", "")[:600]
            prompt = (
                f"You are a financial analyst reading a SEC 10-K table.\n"
                f"Section: {section_heading!r}.{units_note}\n\n"
                f"Raw table text (excerpt):\n{raw_snippet}\n\n"
                f"The table has {n_cols} columns. The automated parser could not identify "
                f"the column headers from this text.\n\n"
                f"Reply with JSON only, no explanation:\n"
                f'{{"columns": ["<col1>", "<col2>", ...], "summary": "<1-2 sentences>"}}\n\n'
                f"Rules:\n"
                f"- Return exactly {n_cols} column names inferred from the raw text above.\n"
                f"- If you cannot determine a column name, use null for that entry.\n"
                f"- summary: name the metric(s), time periods, and any notable trend. "
                f"Preserve numeric values exactly. Do not invent information."
            )
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_response = resp.choices[0].message.content.strip()
            parsed_resp = json.loads(raw_response)
            summary = parsed_resp.get("summary", "")
            inferred_cols = parsed_resp.get("columns", [])
            if isinstance(inferred_cols, list) and len(inferred_cols) == n_cols:
                inferred_cols = [c if c and c != "null" else f"Col{i+1}" for i, c in enumerate(inferred_cols)]
            else:
                inferred_cols = []

            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
            estimated_cost = _estimate_table_llm_cost(prompt_tokens, completion_tokens)
            _append_table_llm_metrics({
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": getattr(resp, "model", OPENAI_MODEL),
                "status": "ok_infer",
                "prompt_chars": len(prompt),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "response_chars": len(raw_response),
                "estimated_cost_usd": round(estimated_cost, 8),
                "error": "",
            })
            if not summary:
                return {"summary": fallback_summary, "columns": [], "used_llm": False, "cache_status": "miss"}

            payload_json = json.dumps(_infer_payload(parsed_table, section_heading), ensure_ascii=False, sort_keys=True)
            entry = {
                "cache_version": _TABLE_LLM_SUMMARY_CACHE_VERSION + "_infer",
                "model": getattr(resp, "model", OPENAI_MODEL),
                "payload_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
                "prompt_chars": len(prompt),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(estimated_cost, 8),
                "summary": summary,
                "columns": inferred_cols,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            async with _cache_write_lock:
                cache = _load_table_summary_cache()
                cache[cache_key] = entry
                _write_table_summary_cache(cache)
            return {"summary": summary, "columns": inferred_cols, "used_llm": True, "cache_status": "miss"}

        except Exception as e:
            _append_table_llm_metrics({
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": OPENAI_MODEL, "status": "error_infer",
                "prompt_chars": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "response_chars": 0, "estimated_cost_usd": 0.0, "error": str(e),
            })
            logger.warning("Async LLM column inference failed: %s", e)
            return {"summary": fallback_summary, "columns": [], "used_llm": False, "cache_status": "miss"}


async def _apply_summary_async(
    parsed: dict,
    chunk: dict,
    client,
    semaphore: asyncio.Semaphore,
) -> None:
    """Async version of _apply_summary."""
    parsed["summary_used_llm"] = False
    parsed["summary_source"] = "deterministic"
    parsed["summary_cache_status"] = ""

    heading = chunk.get("section_heading", "")

    if _has_generic_columns(parsed["columns"]):
        result = await _llm_infer_columns_async(parsed, heading, client, semaphore)
        if result["used_llm"]:
            if result.get("columns"):
                _remap_columns(parsed, result["columns"])
            parsed["summary"] = result["summary"]
            parsed["summary_used_llm"] = True
            parsed["summary_source"] = "llm_inferred"
            parsed["summary_cache_status"] = result["cache_status"]
    else:
        result = await _llm_table_summary_async(parsed, heading, client, semaphore)
        if result["used_llm"]:
            parsed["summary"] = result["summary"]
            parsed["summary_used_llm"] = True
            parsed["summary_source"] = "llm"
            parsed["summary_cache_status"] = result["cache_status"]


async def parse_tables_batch(
    chunks: list[dict],
    use_llm_summaries: bool = True,
    max_concurrent: int = 10,
) -> None:
    """Detect and parse tables for all chunks concurrently, mutating in place.

    In a Jupyter notebook:
        from src.table_parser import parse_tables_batch
        await parse_tables_batch(all_chunks, use_llm_summaries=True)
    """
    from openai import AsyncOpenAI
    from src.constants import OPENAI_API_KEY
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    def _pbar(iterable, **kwargs):
        return tqdm(iterable, **kwargs) if tqdm else iterable

    # Run sync table detection first (CPU-bound, fast)
    for chunk in _pbar(chunks, desc="detecting tables"):
        chunk["tables"] = detect_and_parse_tables(chunk, use_llm_summaries=False)

    if not use_llm_summaries:
        return

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Flatten to (table, chunk) pairs that need LLM calls, skipping cache hits upfront
    cache = _load_table_summary_cache()
    pairs = []
    cached_count = 0
    for chunk in chunks:
        for table in chunk.get("tables", []):
            if not table.get("columns"):
                continue
            if _has_generic_columns(table["columns"]):
                key = _infer_cache_key(table, chunk.get("section_heading", ""), OPENAI_MODEL)
            else:
                key = _table_summary_cache_key(table, chunk.get("section_heading", ""), OPENAI_MODEL)
            if key not in cache:
                pairs.append((table, chunk))
            else:
                cached_count += 1

    logger.info("parse_tables_batch: %d tables need LLM calls (%d already cached)",
                len(pairs), cached_count)

    pbar = tqdm(total=len(pairs), desc=f"LLM summaries (0 cached)") if tqdm else None
    completed = 0

    async def process(table, chunk):
        nonlocal completed
        await _apply_summary_async(table, chunk, client, semaphore)
        completed += 1
        if pbar:
            pbar.update(1)

    await asyncio.gather(*[process(t, c) for t, c in pairs])

    if pbar:
        pbar.set_description(f"LLM summaries ({cached_count} cached, {completed} fetched)")
        pbar.close()

    # Apply cached results for tables we skipped above
    for chunk in chunks:
        for table in chunk.get("tables", []):
            if not table.get("summary_used_llm") and table.get("columns"):
                _apply_summary(table, chunk, use_llm_summaries=True)
