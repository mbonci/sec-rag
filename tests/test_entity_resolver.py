"""Unit tests for entity_resolver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entity_resolver import (
    _normalize,
    _normalize_person,
    _pick_canonical,
    attach_resolved_entities,
    resolve_entities,
)


def _make_chunks(mentions: list[tuple[str, str]]) -> list[dict]:
    """Build minimal chunk dicts with the given (text, entity_type) mentions."""
    entities = [{"text": t, "entity_type": et} for t, et in mentions]
    return [{"entities": entities, "section_heading": "ITEM 1. BUSINESS", "resolved_entities": []}]


# ── normalization ─────────────────────────────────────────────────────────────

def test_normalize_strips_suffix():
    assert _normalize("Apple Inc.") == "apple"
    assert _normalize("Apple Incorporated") == "apple"
    assert _normalize("Microsoft Corp.") == "microsoft"
    assert _normalize("Alphabet LLC") == "alphabet"
    assert _normalize("Example GmbH") == "example"
    assert _normalize("Example AG") == "example"


def test_normalize_strips_punctuation():
    assert _normalize("A.F.L.A.C.") == "a f l a c"


def test_normalize_person_strips_title_and_suffix():
    assert _normalize_person("Mr. Michael I. Roth Jr.") == "michael i roth"


# ── resolution ────────────────────────────────────────────────────────────────

def test_obvious_variants_grouped():
    chunks = _make_chunks([
        ("Apple Inc.", "company"),
        ("Apple", "company"),
        ("APPLE INC", "company"),
    ])
    groups = resolve_entities(chunks)
    company_groups = [g for g in groups if g["entity_type"] == "company"]
    assert len(company_groups) == 1, f"Expected 1 group, got {len(company_groups)}: {company_groups}"


def test_unrelated_entities_not_merged():
    chunks = _make_chunks([
        ("Microsoft", "company"),
        ("Apple", "company"),
        ("Google", "company"),
    ])
    groups = resolve_entities(chunks)
    company_groups = [g for g in groups if g["entity_type"] == "company"]
    assert len(company_groups) == 3, f"Expected 3 groups, got {len(company_groups)}"


def test_canonical_name_is_longest():
    from collections import Counter
    members = ["Apple", "Apple Inc.", "APPLE INC"]
    counts = Counter({"Apple": 3, "Apple Inc.": 1, "APPLE INC": 2})
    canonical = _pick_canonical(members, counts)
    # "Apple Inc." is longest non-all-caps → should win
    assert canonical == "Apple Inc.", f"Got {canonical}"


def test_resolve_entities_returns_list():
    chunks = _make_chunks([("Aflac", "company"), ("Aflac Incorporated", "company")])
    result = resolve_entities(chunks)
    assert isinstance(result, list)
    assert all("canonical_name" in g for g in result)


def test_acronym_variant_grouped():
    chunks = _make_chunks([
        ("The Interpublic Group of Companies, Inc.", "company"),
        ("Interpublic Group", "company"),
        ("IPG", "company"),
    ])
    groups = [g for g in resolve_entities(chunks) if g["entity_type"] == "company"]
    assert len(groups) == 1
    assert "IPG" in groups[0]["variants"]
    assert any(e["resolution_method"] == "acronym" for e in groups[0]["variant_evidence"])


def test_person_variants_grouped_conservatively():
    chunks = _make_chunks([
        ("Michael I. Roth", "person"),
        ("Michael Roth", "person"),
        ("M. I. Roth", "person"),
        ("Mr. Roth", "person"),
    ])
    groups = [g for g in resolve_entities(chunks) if g["entity_type"] == "person"]
    grouped = [set(g["variants"]) for g in groups]
    assert {"Michael I. Roth", "Michael Roth", "M. I. Roth"} in grouped
    assert {"Mr. Roth"} in grouped


def test_company_containment_does_not_merge_meaningful_entities():
    chunks = _make_chunks([
        ("McCann", "company"),
        ("McCann Health", "company"),
        ("MRM McCann", "company"),
        ("Fastenal", "company"),
        ("Fastenal School of Business", "company"),
    ])
    groups = [g for g in resolve_entities(chunks) if g["entity_type"] == "company"]
    assert len(groups) == 5


def test_lookalike_acronyms_do_not_merge():
    chunks = _make_chunks([
        ("PECO", "company"),
        ("Pepco", "company"),
        ("Dominion Energy", "company"),
        ("Dominion Energy Gas", "company"),
    ])
    groups = [g for g in resolve_entities(chunks) if g["entity_type"] == "company"]
    assert len(groups) == 4


def test_standalone_legal_suffixes_do_not_collapse():
    chunks = _make_chunks([
        ("LLC", "company"),
        ("Inc.", "company"),
        ("Corporation", "company"),
    ])
    groups = [g for g in resolve_entities(chunks) if g["entity_type"] == "company"]
    assert len(groups) == 3


def test_attach_resolved_entities_includes_variant_provenance():
    chunks = _make_chunks([
        ("Aflac", "company"),
        ("Aflac Incorporated", "company"),
    ])
    groups = resolve_entities(chunks)
    attached = attach_resolved_entities(chunks, groups)
    resolved = attached[0]["resolved_entities"]
    assert resolved
    assert all("resolution_score" in item for item in resolved)


def test_person_entities_separated_from_company():
    chunks = _make_chunks([
        ("Apple", "company"),
        ("Tim Cook", "person"),
    ])
    groups = resolve_entities(chunks)
    types = {g["entity_type"] for g in groups}
    assert "company" in types
    assert "person" in types


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
