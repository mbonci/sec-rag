"""Unit tests for table_parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import table_parser as tp
from src.table_parser import (
    _find_pipe_blocks,
    _parse_pipe_table,
    _parse_space_table,
    detect_and_parse_tables,
)

# ── space-aligned tables ──────────────────────────────────────────────────────

_SPACE_TABLE = """\
(In millions, except per-share amounts)   2014   2013   2012

Net premiums    $ 19,072   $ 20,135   $ 22,148
Net investment income   3,319   3,293   3,473
Total revenues  22,728   23,939   25,364
"""

_SPACE_TABLE_NO_UNITS = """\
Year    Revenue    Net Income
2023    10,200     1,400
2022    9,800      1,200
2021    8,500      900
"""


def test_space_table_detected():
    parsed = _parse_space_table(_SPACE_TABLE, units_hint="(In millions, except per-share amounts)")
    assert parsed is not None, "Expected a table to be parsed"
    assert parsed["rows"], "Expected rows to be non-empty"


def test_space_table_has_summary():
    parsed = _parse_space_table(_SPACE_TABLE, units_hint="(In millions)")
    assert parsed is not None
    assert "summary" in parsed
    assert len(parsed["summary"]) > 0


def test_space_table_columns_detected():
    parsed = _parse_space_table(_SPACE_TABLE, units_hint="(In millions)")
    assert parsed is not None
    # Should detect year columns 2014, 2013, 2012
    cols = parsed.get("columns", [])
    assert any("2014" in c or "2013" in c or "2012" in c for c in cols), f"No year columns in {cols}"


# ── pipe-separated tables ─────────────────────────────────────────────────────

_PIPE_TABLE = """\
| Year | Revenue | Net Income |
|------|---------|------------|
| 2023 | $10.2B  | $1.4B      |
| 2022 | $9.8B   | $1.2B      |
"""


def test_pipe_table_detected():
    blocks = _find_pipe_blocks(_PIPE_TABLE)
    assert len(blocks) >= 1


def test_pipe_table_parsed():
    blocks = _find_pipe_blocks(_PIPE_TABLE)
    assert blocks
    parsed = _parse_pipe_table(blocks[0])
    assert parsed is not None
    assert parsed["columns"] == ["Year", "Revenue", "Net Income"]
    assert len(parsed["rows"]) == 2


def test_pipe_table_row_values():
    blocks = _find_pipe_blocks(_PIPE_TABLE)
    parsed = _parse_pipe_table(blocks[0])
    assert parsed is not None
    first_row = parsed["rows"][0]
    assert isinstance(first_row, dict)
    assert first_row["Year"] == "2023"


# ── detect_and_parse_tables integration ──────────────────────────────────────

def test_detect_tables_in_chunk():
    chunk = {
        "section_heading": "ITEM 6. SELECTED FINANCIAL DATA",
        "section_text": _SPACE_TABLE,
    }
    tables = detect_and_parse_tables(chunk)
    assert len(tables) >= 1


def test_no_table_in_plain_text():
    chunk = {
        "section_heading": "ITEM 1. BUSINESS",
        "section_text": "This is a paragraph of plain text with no table structure whatsoever.",
    }
    tables = detect_and_parse_tables(chunk)
    assert len(tables) == 0


def test_llm_table_summary_cache_reuses_successful_result(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "_TABLE_LLM_SUMMARY_CACHE_JSON", tmp_path / "table_summary_cache.json")
    monkeypatch.setattr(tp, "_TABLE_LLM_SUMMARY_METRICS_CSV", tmp_path / "table_summary_metrics.csv")
    monkeypatch.setattr(tp, "OPENAI_MODEL", "test-model")

    calls = []

    def fake_uncached(parsed_table: dict, section_heading: str) -> dict:
        calls.append(section_heading)
        return {
            "summary": "LLM summary.",
            "model": "test-model",
            "prompt_chars": 100,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "estimated_cost_usd": 0.00001,
        }

    monkeypatch.setattr(tp, "_llm_table_summary_uncached", fake_uncached)
    parsed = _parse_space_table(_SPACE_TABLE, units_hint="(In millions)")
    assert parsed is not None

    first = tp._llm_table_summary_cached(parsed, "ITEM 6. SELECTED FINANCIAL DATA")
    second = tp._llm_table_summary_cached(parsed, "ITEM 6. SELECTED FINANCIAL DATA")

    assert first["summary"] == "LLM summary."
    assert second["summary"] == "LLM summary."
    assert second["cache_status"] == "hit"
    assert len(calls) == 1


def test_llm_table_summary_cache_misses_when_table_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "_TABLE_LLM_SUMMARY_CACHE_JSON", tmp_path / "table_summary_cache.json")
    monkeypatch.setattr(tp, "_TABLE_LLM_SUMMARY_METRICS_CSV", tmp_path / "table_summary_metrics.csv")
    monkeypatch.setattr(tp, "OPENAI_MODEL", "test-model")

    calls = []

    def fake_uncached(parsed_table: dict, section_heading: str) -> dict:
        calls.append(parsed_table["rows"][0])
        return {"summary": f"Summary {len(calls)}.", "model": "test-model"}

    monkeypatch.setattr(tp, "_llm_table_summary_uncached", fake_uncached)
    parsed_a = _parse_space_table(_SPACE_TABLE, units_hint="(In millions)")
    parsed_b = _parse_space_table(_SPACE_TABLE.replace("19,072", "19,999"), units_hint="(In millions)")
    assert parsed_a is not None
    assert parsed_b is not None

    tp._llm_table_summary_cached(parsed_a, "ITEM 6. SELECTED FINANCIAL DATA")
    tp._llm_table_summary_cached(parsed_b, "ITEM 6. SELECTED FINANCIAL DATA")

    assert len(calls) == 2


def test_detect_tables_marks_llm_summary_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "_TABLE_LLM_SUMMARY_CACHE_JSON", tmp_path / "table_summary_cache.json")
    monkeypatch.setattr(tp, "_TABLE_LLM_SUMMARY_METRICS_CSV", tmp_path / "table_summary_metrics.csv")
    monkeypatch.setattr(tp, "OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        tp,
        "_llm_table_summary_uncached",
        lambda parsed_table, section_heading: {"summary": "LLM summary.", "model": "test-model"},
    )
    chunk = {
        "section_heading": "ITEM 6. SELECTED FINANCIAL DATA",
        "section_text": _SPACE_TABLE,
    }

    tables = detect_and_parse_tables(chunk, use_llm_summaries=True)

    assert tables
    assert tables[0]["summary"] == "LLM summary."
    assert tables[0]["summary_used_llm"] is True
    assert tables[0]["summary_source"] == "llm"
    assert tables[0]["summary_cache_status"] == "miss"


def test_detect_tables_marks_deterministic_summary_provenance():
    chunk = {
        "section_heading": "ITEM 6. SELECTED FINANCIAL DATA",
        "section_text": _SPACE_TABLE,
    }

    tables = detect_and_parse_tables(chunk, use_llm_summaries=False)

    assert tables
    assert tables[0]["summary_used_llm"] is False
    assert tables[0]["summary_source"] == "deterministic"


def test_monetary_regex_extraction():
    from src.entity_extractor import _extract_monetary_values
    text = "Revenue was $1.2 million and net income was USD 500,000."
    vals = _extract_monetary_values(text)
    assert len(vals) >= 2
    currencies = {v["currency"] for v in vals}
    assert "USD" in currencies


def test_monetary_regex_handles_magnitude():
    from src.entity_extractor import _extract_monetary_values
    matches = _extract_monetary_values("$4.5 billion")
    assert matches
    assert matches[0]["amount"] == 4_500_000_000


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
