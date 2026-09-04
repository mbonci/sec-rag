"""Rule-based section detection and categorization.

Strategy:
1. Extract the table of contents to discover which Item numbers appear in this
   document and where the TOC block ends.
2. After the TOC, search for each item's body heading using a flexible per-item
   regex that handles all observed formatting variants (uppercase/mixed case,
   period vs space delimiter, inline vs standalone heading lines, non-breaking
   spaces).
3. Map sections to categories via item number first (deterministic per SEC
   rules), falling back to title-text regex for anything not in the map.
4. Fall back to the original broad regex scan if no usable TOC is found.
"""

import logging
import csv
import hashlib
import json
import re
import time
from typing import Optional

from src.constants import (
    CACHE_DIR,
    COSTS_DIR,
    HEADING_CATEGORY_RULES,
    ITEM_CATEGORY_MAP,
    OPENAI_MODEL,
)
from src.utils import make_document_id, make_section_id, normalize_whitespace

logger = logging.getLogger(__name__)

# Matches "Table of Contents" / "TABLE OF CONTENTS" variants
_TOC_HEADER_RE = re.compile(r"table[\s\xa0]+of[\s\xa0]+contents", re.IGNORECASE)

# Matches an Item reference: "Item 1", "ITEM 1A", "Item\xa01B" etc.
# Captures just the item key (digits + optional letter suffix).
_TOC_ITEM_RE = re.compile(r"\bitem[\s\xa0]+(\d+[A-Za-z]?)\b", re.IGNORECASE)

# Page separator lines produced by SEC text conversion
_PAGE_SEP_RE = re.compile(r"^-{20,}\s*$", re.MULTILINE)

# Minimum items to consider a block a real TOC (vs. boilerplate)
_MIN_TOC_ITEMS = 4

# How far into the document to search for the TOC header (chars)
_TOC_SEARCH_LIMIT = 120_000

# Valid 10-K item key: 1-2 digits + optional single letter (rejects "401" etc.)
_VALID_ITEM_RE = re.compile(r"^\d{1,2}[A-Za-z]?$")

# If the average char-gap between consecutive TOC item matches exceeds this,
# the items are likely scattered body headings, not a compact TOC listing.
_MAX_TOC_AVG_GAP = 400

# gpt-4.1-mini pricing estimate (USD per 1M tokens)
# Keep these together so the cost model is easy to update if pricing changes.
_GPT41_MINI_INPUT_COST_PER_1M_TOKENS = 0.40
_GPT41_MINI_OUTPUT_COST_PER_1M_TOKENS = 1.60

_TOC_LLM_METRICS_CSV = COSTS_DIR / "toc_llm_metrics.csv"
_TOC_LLM_CACHE_JSON = CACHE_DIR / "toc_llm_cache.json"
_TOC_LLM_CACHE_VERSION = "toc_llm_v1"


# ── TOC extraction ────────────────────────────────────────────────────────────

def extract_toc(text: str) -> tuple[list[str], int]:
    """Find the TOC and return (ordered_item_keys, toc_end_char_pos).

    Searches for ALL "Table of Contents" occurrences up to _TOC_SEARCH_LIMIT
    and picks the one whose following 3000 chars contains the most Item
    references. Returns ([], 0) if no usable TOC found.
    """
    result = _extract_toc_with_metadata(text)
    return result["items"], result["toc_end"]


def _extract_toc_with_metadata(
    text: str,
    doc_id: str = "",
    source_path: str = "",
) -> dict:
    """Find the TOC and include provenance for the chosen strategy."""
    search_space = text[:_TOC_SEARCH_LIMIT]
    best_items: list[str] = []
    best_end: int = 0
    method = "toc"

    for toc_m in _TOC_HEADER_RE.finditer(search_space):
        toc_start = toc_m.start()
        block = text[toc_start: toc_start + 8000]

        items: list[str] = []
        seen: set[str] = set()
        positions: list[int] = []
        last_item_end = 0

        for m in _TOC_ITEM_RE.finditer(block):
            key = m.group(1).upper()
            # Reject invalid item numbers (e.g. "401" from Regulation S-K refs)
            if not _VALID_ITEM_RE.match(key):
                continue
            if key not in seen:
                seen.add(key)
                items.append(key)
                positions.append(m.start())
                last_item_end = m.end()

        # Reject block if consecutive items are too far apart on average —
        # that pattern indicates scattered body headings, not a compact TOC.
        if len(positions) >= 2:
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            if sum(gaps) / len(gaps) > _MAX_TOC_AVG_GAP:
                continue

        if len(items) > len(best_items):
            best_items = items
            # End of TOC = first long separator after the last item, or a fixed offset
            after_items = block[last_item_end:]
            sep_m = re.search(r"-{20,}", after_items)
            if sep_m:
                best_end = toc_start + last_item_end + sep_m.end()
            else:
                best_end = toc_start + last_item_end + min(500, len(after_items))

    # Secondary pass: headerless cluster scan for documents with no usable TOC
    # header (e.g. TOC listed without "Table of Contents" prefix, or formats
    # where the header is in a non-standard location).
    # Only runs if the header-guided pass failed.
    if len(best_items) < _MIN_TOC_ITEMS:
        best_items, best_end = _extract_toc_headerless(text)
        method = "headerless_toc"

    if len(best_items) < _MIN_TOC_ITEMS:
        llm_result = _extract_toc_via_llm_cached(text, doc_id=doc_id, source_path=source_path)
        best_items = llm_result["items"]
        best_end = 0  # body search starts from beginning when LLM supplies TOC
        method = "llm_toc"

    if len(best_items) < _MIN_TOC_ITEMS:
        return {
            "items": [],
            "toc_end": 0,
            "method": "none",
            "used_llm": False,
        }

    return {
        "items": best_items,
        "toc_end": best_end,
        "method": method,
        "used_llm": method == "llm_toc",
    }


# How many chars to send to the LLM when the TOC header position is unknown
_LLM_TOC_CHARS_DEFAULT = 8_000
# Window size to send when we know where the TOC header is
_LLM_TOC_WINDOW = 5_000


def _estimate_toc_llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate gpt-4.1-mini cost in USD for a single call."""
    return (
        (prompt_tokens / 1_000_000) * _GPT41_MINI_INPUT_COST_PER_1M_TOKENS
        + (completion_tokens / 1_000_000) * _GPT41_MINI_OUTPUT_COST_PER_1M_TOKENS
    )


def _append_toc_llm_metrics(row: dict) -> None:
    """Append one TOC LLM metrics record to the CSV sidecar."""
    _TOC_LLM_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = _TOC_LLM_METRICS_CSV.exists() and _TOC_LLM_METRICS_CSV.stat().st_size > 0

    fieldnames = [
        "timestamp_utc",
        "model",
        "status",
        "snippet_source",
        "prompt_chars",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
        "response_chars",
        "items_returned",
        "finish_reason",
        "estimated_cost_usd",
        "error",
    ]

    with _TOC_LLM_METRICS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _select_toc_llm_snippet(text: str) -> tuple[str, str]:
    """Return the focused text snippet sent to the TOC LLM fallback."""
    toc_m = _TOC_HEADER_RE.search(text[:_TOC_SEARCH_LIMIT])
    if toc_m:
        return text[toc_m.start(): toc_m.start() + _LLM_TOC_WINDOW], "toc_header"
    return text[:_LLM_TOC_CHARS_DEFAULT], "document_start"


def _load_toc_llm_cache() -> dict:
    if not _TOC_LLM_CACHE_JSON.exists():
        return {}
    try:
        return json.loads(_TOC_LLM_CACHE_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Ignoring unreadable TOC LLM cache %s: %s", _TOC_LLM_CACHE_JSON, e)
        return {}


def _write_toc_llm_cache(cache: dict) -> None:
    _TOC_LLM_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    _TOC_LLM_CACHE_JSON.write_text(
        json.dumps(cache, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _toc_llm_cache_key(
    doc_id: str,
    source_path: str,
    model: str,
    snippet_source: str,
    snippet: str,
) -> str:
    snippet_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
    raw = json.dumps({
        "cache_version": _TOC_LLM_CACHE_VERSION,
        "document_id": doc_id,
        "source_path": source_path,
        "model": model,
        "snippet_source": snippet_source,
        "snippet_sha256": snippet_hash,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _append_toc_llm_cache_hit_metrics(entry: dict) -> None:
    _append_toc_llm_metrics({
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": entry.get("model", OPENAI_MODEL),
        "status": "cache_hit",
        "snippet_source": entry.get("snippet_source", ""),
        "prompt_chars": entry.get("prompt_chars", 0),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0,
        "response_chars": 0,
        "items_returned": len(entry.get("items", [])),
        "finish_reason": "",
        "estimated_cost_usd": 0.0,
        "error": "",
    })


def _extract_toc_via_llm(text: str) -> list[str]:
    """Backward-compatible wrapper returning only item keys."""
    return _extract_toc_via_llm_cached(text)["items"]


def _extract_toc_via_llm_cached(
    text: str,
    doc_id: str = "",
    source_path: str = "",
) -> dict:
    """Return LLM TOC items, using a content-addressed cache when possible."""
    snippet, snippet_source = _select_toc_llm_snippet(text)
    cache_key = _toc_llm_cache_key(doc_id, source_path, OPENAI_MODEL, snippet_source, snippet)
    cache = _load_toc_llm_cache()

    if cache_key in cache:
        entry = cache[cache_key]
        cached_items = entry.get("items", [])
        _append_toc_llm_cache_hit_metrics(entry)
        logger.info("LLM TOC cache hit for %s (%d items)", source_path or doc_id, len(cached_items))
        return {"items": cached_items, "cache_status": "hit"}

    result = _extract_toc_via_llm_uncached(snippet, snippet_source)
    items = result.get("items", [])
    # Cache all results including empty ones — prevents re-querying the LLM on every run
    # for documents whose TOC couldn't be found in the selected snippet.
    cache[cache_key] = {
        "cache_version": _TOC_LLM_CACHE_VERSION,
        "document_id": doc_id,
        "source_path": source_path,
        "model": result.get("model", OPENAI_MODEL),
        "snippet_source": snippet_source,
        "snippet_sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
        "prompt_chars": len(snippet),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "estimated_cost_usd": result.get("estimated_cost_usd", 0.0),
        "items": items,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_toc_llm_cache(cache)

    return {"items": items, "cache_status": "miss"}


def _extract_toc_via_llm_uncached(snippet: str, snippet_source: str) -> dict:
    """Call OpenAI to extract ordered Item numbers from a focused snippet."""
    started_at = time.perf_counter()
    try:
        from openai import OpenAI
        from src.constants import OPENAI_API_KEY, OPENAI_MODEL

        if not OPENAI_API_KEY:
            logger.debug("LLM TOC extraction skipped: no API key")
            return {"items": []}

        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt_chars = len(snippet)
        prompt = (
            "The text below is the beginning of an SEC 10-K annual report filing. "
            "Your task is to extract the list of Item numbers that appear in its "
            "Table of Contents.\n\n"
            "Return ONLY a JSON array of item-key strings in document order, "
            "e.g. [\"1\", \"1A\", \"1B\", \"2\", \"3\", \"4\", \"5\", \"6\", "
            "\"7\", \"7A\", \"8\", \"9\", \"9A\", \"9B\", \"10\", \"11\", "
            "\"12\", \"13\", \"14\", \"15\"]. "
            "Include only items that explicitly appear in this document's TOC. "
            "Do not add items that are not listed. Output nothing except the JSON array.\n\n"
            f"DOCUMENT START:\n{snippet}"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
        )
        import json
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("Expected a JSON array")
        # Normalise and validate each key
        result = [str(k).upper() for k in items if _VALID_ITEM_RE.match(str(k).upper())]
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response_chars = len(raw)
        estimated_cost = _estimate_toc_llm_cost(prompt_tokens, completion_tokens)
        finish_reason = getattr(resp.choices[0], "finish_reason", "") or ""
        _append_toc_llm_metrics({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": getattr(resp, "model", OPENAI_MODEL),
            "status": "ok",
            "snippet_source": snippet_source,
            "prompt_chars": prompt_chars,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "response_chars": response_chars,
            "items_returned": len(result),
            "finish_reason": finish_reason,
            "estimated_cost_usd": round(estimated_cost, 8),
            "error": "",
        })
        logger.info("LLM TOC extraction returned %d items: %s", len(result), result)
        return {
            "items": result,
            "model": getattr(resp, "model", OPENAI_MODEL),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _append_toc_llm_metrics({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "gpt-4.1-mini",
            "status": "error",
            "snippet_source": "",
            "prompt_chars": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": latency_ms,
            "response_chars": 0,
            "items_returned": 0,
            "finish_reason": "",
            "estimated_cost_usd": 0.0,
            "error": str(e),
        })
        logger.warning("LLM TOC extraction failed: %s", e)
        return {"items": []}


# Standard 10-K Item numbers used to augment a partial TOC when coverage is poor.
_STANDARD_10K_ITEMS = [
    '1', '1A', '1B', '2', '3', '4', '5', '6',
    '7', '7A', '8', '9', '9A', '9B',
    '10', '11', '12', '13', '14', '15',
]
# If the last body heading is before this fraction of the document, the TOC is
# probably incomplete (inline cross-references mistaken for a TOC listing).
_MIN_COVERAGE_RATIO = 0.70

# Maximum char distance between consecutive items to be in the same cluster
_MAX_ITEM_GAP = 800
# How far into the document to scan for a headerless TOC cluster
_HEADERLESS_SCAN_LIMIT = 50_000
# Stricter avg-gap threshold for headerless scan (no TOC header anchor),
# since we have no external signal that we're inside a real TOC block.
# Expeditors genuine TOC: avg_gap≈58. Ohio false body-heading cluster: avg_gap≈243.
_MAX_TOC_AVG_GAP_HEADERLESS = 150


def _extract_toc_headerless(text: str) -> tuple[list[str], int]:
    """Find a dense Item cluster without requiring a 'Table of Contents' header.

    Scans the first _HEADERLESS_SCAN_LIMIT chars for the tightest cluster of
    distinct Item references. Returns ([], 0) if no qualifying cluster found.
    """
    scan = text[:_HEADERLESS_SCAN_LIMIT]
    matches = [
        (m.start(), m.group(1).upper())
        for m in _TOC_ITEM_RE.finditer(scan)
        if _VALID_ITEM_RE.match(m.group(1).upper())
    ]

    best_items: list[str] = []
    best_end: int = 0
    best_avg_gap: float = float("inf")

    i = 0
    while i < len(matches):
        cluster_pos: list[int] = [matches[i][0]]
        cluster_keys: list[str] = [matches[i][1]]
        seen: set[str] = {matches[i][1]}
        j = i + 1
        while j < len(matches) and matches[j][0] - matches[j - 1][0] < _MAX_ITEM_GAP:
            key = matches[j][1]
            if key not in seen:
                seen.add(key)
                cluster_pos.append(matches[j][0])
                cluster_keys.append(key)
            j += 1

        if len(cluster_keys) >= _MIN_TOC_ITEMS:
            gaps = [cluster_pos[k + 1] - cluster_pos[k] for k in range(len(cluster_pos) - 1)]
            avg_gap = sum(gaps) / len(gaps)

            if avg_gap <= _MAX_TOC_AVG_GAP_HEADERLESS and (
                len(cluster_keys) > len(best_items)
                or (len(cluster_keys) == len(best_items) and avg_gap < best_avg_gap)
            ):
                best_items = cluster_keys
                best_avg_gap = avg_gap
                last_pos = cluster_pos[-1]
                after = scan[last_pos:]
                sep_m = re.search(r"-{20,}", after)
                best_end = last_pos + (sep_m.end() if sep_m else min(500, len(after)))

        i = j if j > i else i + 1

    return best_items, best_end


# ── Body heading detection ────────────────────────────────────────────────────

def _item_body_re(item_key: str) -> re.Pattern:
    """Strict regex: Item N at start of line, optional PART-prefix on same line.

    Handles:
      - Leading spaces / tabs / non-breaking spaces
      - Optional "PART I." or "PART II." prefix on the same line
      - Period, colon, space, non-breaking space, em dash, or en dash after key
      - Non-breaking spaces inside "Item N"
      - Negative lookahead (?!\\d) to avoid matching decimal form refs like
        "Item 5.05" (Form 8-K cross-references) as "Item 5"
    """
    esc = re.escape(item_key)
    part_opt = r"(?:part[\s\xa0]+\S+[\s\xa0]+)?"
    delim = r"[.:\s\xa0–—]"
    return re.compile(
        rf"(?im)^[ \t\xa0]*{part_opt}item[\s\xa0]+{esc}{delim}(?!\d)",
    )


def _item_loose_re(item_key: str) -> re.Pattern:
    """Loose regex: Item N anywhere (for sequential window search).

    Same negative lookahead to avoid decimal cross-references.
    """
    esc = re.escape(item_key)
    delim = r"[.:\s\xa0–—]"
    return re.compile(
        rf"(?i)\bitem[\s\xa0]+{esc}{delim}(?!\d)",
    )


def _extract_heading_text(text: str, match_start: int, match_end: int) -> str:
    """Return the heading string for a match, peeking at the next line if needed."""
    line_end = text.find("\n", match_end)
    line_text = text[match_start: line_end if line_end != -1 else match_end].strip()
    line_text = normalize_whitespace(line_text)

    # If title is absent from the matched line, grab the first non-empty line below
    # (handles Palantir-style "ITEM 8.\n\nFINANCIAL STATEMENTS")
    after = text[line_end + 1: line_end + 300] if line_end != -1 else ""
    stripped_line = re.sub(r"(?i)item[\s\xa0]+\d+[a-z]?[.\s]*", "", line_text).strip()
    if not stripped_line:
        for extra in after.splitlines():
            extra = extra.strip()
            if extra:
                line_text = f"{line_text} {extra}"
                break

    return normalize_whitespace(line_text)


def find_body_headings(
    text: str,
    item_keys: list[str],
    search_from: int,
) -> list[tuple[int, str, str]]:
    """For each item key, find its body heading position in text[search_from:].

    Two-pass strategy:
      Pass 1 — strict: item at line-start (with optional PART prefix, colon).
      Pass 2 — loose: for items still missing, search within the sequential
                window [prev_found_pos, next_found_pos] to handle chained
                inline headings (e.g. "Item 1B. None. Item 2. Properties...").

    Returns list of (abs_char_pos, item_key, heading_text) sorted by position.
    """
    body = text[search_from:]
    found: dict[str, tuple[int, str]] = {}  # key -> (abs_pos, heading)

    # Pass 1: strict line-start match
    for key in item_keys:
        m = _item_body_re(key).search(body)
        if m:
            abs_pos = search_from + m.start()
            heading = _extract_heading_text(text, abs_pos, search_from + m.end())
            found[key] = (abs_pos, heading)

    # Pass 2: sequential window search for items still missing
    missing = [k for k in item_keys if k not in found]
    if missing:
        for key in missing:
            idx = item_keys.index(key)

            # Window: after the previous found key, before the next found key
            prev_pos = search_from
            for prev_key in reversed(item_keys[:idx]):
                if prev_key in found:
                    prev_pos = found[prev_key][0] + 1
                    break

            next_pos = len(text)
            for next_key in item_keys[idx + 1:]:
                if next_key in found:
                    next_pos = found[next_key][0]
                    break

            window = text[prev_pos:next_pos]
            m = _item_loose_re(key).search(window)
            if m:
                abs_pos = prev_pos + m.start()
                heading = _extract_heading_text(text, abs_pos, prev_pos + m.end())
                found[key] = (abs_pos, heading)

    result = [(pos, key, hdg) for key, (pos, hdg) in found.items()]
    result.sort(key=lambda x: x[0])
    return result


# ── Categorization ────────────────────────────────────────────────────────────

def _categorize_by_item_key(item_key: str) -> Optional[str]:
    return ITEM_CATEGORY_MAP.get(item_key.upper())


def _categorize_by_title(heading: str) -> str:
    normalized = heading.lower()
    for pattern, category in HEADING_CATEGORY_RULES:
        if re.search(pattern, normalized):
            return category
    return "other"


def _categorize(item_key: str, heading: str) -> str:
    return _categorize_by_item_key(item_key) or _categorize_by_title(heading)


# ── Section building ──────────────────────────────────────────────────────────

def _make_section(
    doc_id: str,
    source_path: str,
    heading: str,
    category: str,
    body: str,
    parse_method: str = "fallback_regex",
    parse_used_llm: bool = False,
) -> dict:
    return {
        "section_id": make_section_id(source_path, heading),
        "document_id": doc_id,
        "source_path": source_path,
        "section_heading": heading,
        "section_category": category,
        "section_text": body.strip(),
        "section_parse_method": parse_method,
        "section_parse_used_llm": parse_used_llm,
        "entities": [],
        "resolved_entities": [],
        "tables": [],
        "page_numbers": [],
        "processing_warnings": [],
    }


def _clean_body(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(l for l in lines if not _PAGE_SEP_RE.match(l))


def _build_sections_from_headings(
    doc_id: str,
    source_path: str,
    text: str,
    headings: list[tuple[int, str, str]],
    parse_method: str = "toc",
    parse_used_llm: bool = False,
) -> list[dict]:
    """Convert (pos, key, heading_text) triples into section dicts."""
    sections: list[dict] = []

    if not headings:
        return sections

    # Preamble: cover page + TOC block before the first item heading
    pre = _clean_body(text[: headings[0][0]])
    if pre.strip():
        sections.append(_make_section(
            doc_id, source_path, "Preamble", "other", pre,
            parse_method=parse_method, parse_used_llm=parse_used_llm,
        ))

    for i, (pos, key, heading) in enumerate(headings):
        # Body starts on the line after the heading line
        line_end = text.find("\n", pos)
        body_start = line_end + 1 if line_end != -1 else pos
        # Last item body ends at the next heading; final item ends at EOF
        body_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        body = _clean_body(text[body_start:body_end])
        category = _categorize(key, heading)
        sections.append(_make_section(
            doc_id, source_path, heading, category, body,
            parse_method=parse_method, parse_used_llm=parse_used_llm,
        ))

    return sections


# ── Test-facing helpers ───────────────────────────────────────────────────────

def _categorize_heading(heading: str) -> str:
    """Categorize a heading string (item key extracted from text, then mapped)."""
    km = re.search(r"\d+[A-Za-z]?", heading)
    key = km.group(0).upper() if km else ""
    return _categorize(key, heading)


def _is_toc_entry(text: str) -> bool:
    """Return True if text looks like a TOC entry (trailing page number)."""
    return bool(_TOC_TRAILING_NUM_RE.search(text))


# ── Fallback: broad regex scan ────────────────────────────────────────────────

# Legacy regex kept as fallback for documents with no parseable TOC.
# Accepts colon delimiter and allows empty title (title may be on next line).
_ITEM_RE = re.compile(
    r"^(ITEM[\s\xa0]+\d+[A-Z]?[.:\s].{0,120})$",
    re.IGNORECASE | re.MULTILINE,
)
_TOC_TRAILING_NUM_RE = re.compile(r"[\s\xa0]+\d+\s*$")


def _parse_sections_fallback(doc_id: str, source_path: str, text: str) -> list[dict]:
    """Original broad-regex section parser, used when TOC extraction fails."""
    heading_spans: list[tuple[int, int, str]] = []
    for m in _ITEM_RE.finditer(text):
        heading_text = normalize_whitespace(m.group(1))
        if _TOC_TRAILING_NUM_RE.search(m.group(1)):
            continue
        if len(heading_text.split()) < 2:
            continue
        heading_spans.append((m.start(), m.end(), heading_text))

    if not heading_spans:
        return [_make_section(doc_id, source_path, "Document", "other", text)]

    sections: list[dict] = []
    pre = text[: heading_spans[0][0]].strip()
    if pre:
        sections.append(_make_section(doc_id, source_path, "Preamble", "other", pre))

    for i, (start, end, heading) in enumerate(heading_spans):
        body_end = heading_spans[i + 1][0] if i + 1 < len(heading_spans) else len(text)
        body = _clean_body(text[end:body_end])
        # Infer item key from heading for category lookup
        km = re.search(r"\d+[A-Za-z]?", heading)
        key = km.group(0).upper() if km else ""
        category = _categorize(key, heading)
        sections.append(_make_section(doc_id, source_path, heading, category, body))

    return sections


# ── Public API ────────────────────────────────────────────────────────────────

def parse_sections(document: dict) -> list[dict]:
    """Split a document into Item-level sections.

    Uses TOC-guided detection when possible; falls back to broad regex scan.
    """
    if "raw_text" not in document:
        text = document["text"]
    else:
        text = document["raw_text"]
    if "source_path" not in document:
        source_path = document["filepath"]
    else:
        source_path = document["source_path"]
    try: 
        doc_id = document["document_id"]
    except KeyError:
        doc_id = make_document_id(source_path)

    toc_info = _extract_toc_with_metadata(text, doc_id=doc_id, source_path=source_path)
    toc_items = toc_info["items"]
    toc_end = toc_info["toc_end"]
    parse_method = toc_info["method"]
    parse_used_llm = toc_info["used_llm"]

    if toc_items:
        body_headings = find_body_headings(text, toc_items, toc_end)
        if len(body_headings) >= _MIN_TOC_ITEMS:
            last_pos = body_headings[-1][0]
            coverage = last_pos / len(text) if text else 0
            if coverage >= _MIN_COVERAGE_RATIO or len(body_headings) >= 12:
                logger.debug(
                    "%s: TOC-guided — %d items, %d found",
                    source_path, len(toc_items), len(body_headings),
                )
                return _build_sections_from_headings(
                    doc_id,
                    source_path,
                    text,
                    body_headings,
                    parse_method=parse_method,
                    parse_used_llm=parse_used_llm,
                )
            # Coverage too low: TOC was likely incomplete (e.g. inline cross-references
            # caught instead of a real TOC listing). Broaden search to all standard items.
            augmented_items = list(dict.fromkeys(toc_items + _STANDARD_10K_ITEMS))
            augmented_headings = find_body_headings(text, augmented_items, 0)
            if len(augmented_headings) >= len(body_headings):
                logger.debug(
                    "%s: augmented TOC %d→%d headings (coverage was %.0f%%)",
                    source_path, len(body_headings), len(augmented_headings), coverage * 100,
                )
                body_headings = augmented_headings
            if len(body_headings) >= _MIN_TOC_ITEMS:
                augmented_method = "llm_toc" if parse_used_llm else "augmented_toc"
                return _build_sections_from_headings(
                    doc_id,
                    source_path,
                    text,
                    body_headings,
                    parse_method=augmented_method,
                    parse_used_llm=parse_used_llm,
                )
        logger.debug(
            "%s: TOC found %d items but only %d matched in body — falling back",
            source_path, len(toc_items), len(body_headings),
        )

    logger.debug("%s: using fallback regex parser", source_path)
    return _parse_sections_fallback(doc_id, source_path, text)

