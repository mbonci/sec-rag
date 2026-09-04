"""Unit tests for entity_extractor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entity_extractor import _keep_org_entity


def test_keep_org_entity_filters_generic_false_positives():
    for text in ["Company", "LLC", "SEC", "FERC", "FASB"]:
        assert not _keep_org_entity(text), f"Expected {text!r} to be filtered"


def test_keep_org_entity_keeps_real_organizations():
    for text in ["Apple Inc.", "General Electric Company", "The Walt Disney Company"]:
        assert _keep_org_entity(text), f"Expected {text!r} to be kept"
