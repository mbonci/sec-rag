"""Unit tests for section_parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import section_parser as sp
from src.section_parser import (
    _categorize_heading,
    _is_toc_entry,
    parse_sections,
)

# ── heading detection ─────────────────────────────────────────────────────────

_FAKE_DOC_TEMPLATE = {
    "document_id": "test001",
    "source_path": "/fake/test.txt",
    "file_type": "txt",
    "metadata": {},
}


def _make_doc(raw_text: str) -> dict:
    return {**_FAKE_DOC_TEMPLATE, "raw_text": raw_text}


def test_detects_item1_heading(monkeypatch):
    monkeypatch.setattr(sp, "_extract_toc_via_llm_cached", lambda *args, **kwargs: {"items": []})
    doc = _make_doc("ITEM 1. BUSINESS\nThis is the business section.\n\nSome more text.")
    sections = parse_sections(doc)
    headings = [s["section_heading"] for s in sections]
    assert any("ITEM 1" in h for h in headings), f"No ITEM 1 heading found in {headings}"


def test_detects_multiple_items(monkeypatch):
    monkeypatch.setattr(sp, "_extract_toc_via_llm_cached", lambda *args, **kwargs: {"items": []})
    text = (
        "ITEM 1. BUSINESS\nBusiness content here.\n\n"
        "ITEM 1A. RISK FACTORS\nRisk content here.\n\n"
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\nMD&A content.\n"
    )
    sections = parse_sections(_make_doc(text))
    categories = {s["section_category"] for s in sections}
    assert "business_overview" in categories
    assert "risk_factors" in categories
    assert "management_discussion" in categories


def test_skips_toc_entries():
    # TOC entries end with trailing page number
    assert _is_toc_entry("Item 1. Business 1") is True
    assert _is_toc_entry("Item 1A. Risk Factors 12") is True


def test_real_heading_not_toc():
    assert _is_toc_entry("ITEM 1. BUSINESS We prepare our financial statements") is False


def test_no_headings_returns_single_section(monkeypatch):
    monkeypatch.setattr(sp, "_extract_toc_via_llm_cached", lambda *args, **kwargs: {"items": []})
    doc = _make_doc("Just a paragraph of plain text with no headings at all.")
    sections = parse_sections(doc)
    assert len(sections) == 1
    assert sections[0]["section_category"] == "other"


def test_toc_llm_cache_reuses_successful_result(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_TOC_LLM_CACHE_JSON", tmp_path / "toc_llm_cache.json")
    monkeypatch.setattr(sp, "_TOC_LLM_METRICS_CSV", tmp_path / "toc_llm_metrics.csv")
    monkeypatch.setattr(sp, "OPENAI_MODEL", "test-model")

    calls = []

    def fake_uncached(snippet: str, snippet_source: str) -> dict:
        calls.append((snippet, snippet_source))
        return {
            "items": ["1", "1A", "1B", "2"],
            "model": "test-model",
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "estimated_cost_usd": 0.00001,
        }

    monkeypatch.setattr(sp, "_extract_toc_via_llm_uncached", fake_uncached)

    text = "No parseable TOC here, only a difficult filing prefix."
    first = sp._extract_toc_via_llm_cached(text, doc_id="doc001", source_path="/fake/doc.txt")
    second = sp._extract_toc_via_llm_cached(text, doc_id="doc001", source_path="/fake/doc.txt")

    assert first["items"] == ["1", "1A", "1B", "2"]
    assert second["items"] == first["items"]
    assert len(calls) == 1
    assert second["cache_status"] == "hit"


def test_toc_llm_cache_misses_when_text_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_TOC_LLM_CACHE_JSON", tmp_path / "toc_llm_cache.json")
    monkeypatch.setattr(sp, "_TOC_LLM_METRICS_CSV", tmp_path / "toc_llm_metrics.csv")
    monkeypatch.setattr(sp, "OPENAI_MODEL", "test-model")

    calls = []

    def fake_uncached(snippet: str, snippet_source: str) -> dict:
        calls.append(snippet)
        return {"items": ["1", "1A", "1B", "2"], "model": "test-model"}

    monkeypatch.setattr(sp, "_extract_toc_via_llm_uncached", fake_uncached)

    sp._extract_toc_via_llm_cached("difficult filing prefix", doc_id="doc001", source_path="/fake/doc.txt")
    sp._extract_toc_via_llm_cached("changed difficult filing prefix", doc_id="doc001", source_path="/fake/doc.txt")

    assert len(calls) == 2


def test_sections_mark_llm_section_parse(monkeypatch):
    def fake_toc_metadata(text: str, doc_id: str = "", source_path: str = "") -> dict:
        return {
            "items": ["1", "1A", "1B", "2"],
            "toc_end": 0,
            "method": "llm_toc",
            "used_llm": True,
        }

    monkeypatch.setattr(sp, "_extract_toc_with_metadata", fake_toc_metadata)
    text = (
        "ITEM 1. BUSINESS\nBusiness content.\n\n"
        "ITEM 1A. RISK FACTORS\nRisk content.\n\n"
        "ITEM 1B. UNRESOLVED STAFF COMMENTS\nNone.\n\n"
        "ITEM 2. PROPERTIES\nProperty content.\n"
    )

    sections = parse_sections(_make_doc(text))

    assert sections
    assert all(s["section_parse_used_llm"] is True for s in sections)
    assert all(s["section_parse_method"] == "llm_toc" for s in sections)


def test_sections_mark_deterministic_section_parse(monkeypatch):
    monkeypatch.setattr(sp, "_extract_toc_via_llm_cached", lambda *args, **kwargs: {"items": []})
    text = (
        "ITEM 1. BUSINESS\nBusiness content.\n\n"
        "ITEM 1A. RISK FACTORS\nRisk content.\n\n"
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\nMD&A content.\n"
    )

    sections = parse_sections(_make_doc(text))

    assert sections
    assert all(s["section_parse_used_llm"] is False for s in sections)
    assert all("section_parse_method" in s for s in sections)


# ── category mapping ──────────────────────────────────────────────────────────

def test_category_risk_factors():
    assert _categorize_heading("ITEM 1A. RISK FACTORS") == "risk_factors"


def test_category_business():
    assert _categorize_heading("ITEM 1. BUSINESS") == "business_overview"


def test_category_mda():
    assert _categorize_heading("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS") == "management_discussion"


def test_category_financial_statements():
    assert _categorize_heading("ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA") == "financial_results"


def test_category_legal():
    assert _categorize_heading("ITEM 3. LEGAL PROCEEDINGS") == "legal_proceedings"


def test_category_governance():
    assert _categorize_heading("ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE") == "governance"


def test_category_unknown_falls_to_other():
    assert _categorize_heading("ITEM 99. SOMETHING UNKNOWN") == "other"


def test_item4_mine_safety_is_disclosures():
    assert _categorize_heading("ITEM 4. MINE SAFETY DISCLOSURES") == "disclosures"


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
