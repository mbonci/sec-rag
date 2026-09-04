"""Entity extraction: companies/orgs, people, monetary values.

Uses spaCy NER for ORG and PERSON, regex for monetary values.
spaCy model is loaded once and reused across calls.


NOTES FOR IMPROVEMENT:
- Send a small random sample of chunks to openai for entity extraction and use it to improve the entity extraction pipeline
    e.g., - what are we missing? what are we getting wrong? what are we getting right?
- Consider using a more powerful spaCy model (e.g., en_core_web_trf)
"""

import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ── monetary value patterns ───────────────────────────────────────────────────

_CURRENCY_SYMBOLS = r"(?:\$|€|£|¥|CHF|USD|EUR|GBP|JPY)"
_MAGNITUDE = r"(?:\s*(?:thousand|million|billion|trillion)s?)"
_NUMBER = r"\d[\d,]*(?:\.\d+)?"

# Matches: $1.2 million, USD 4.5 billion, €300,000, CHF 25 million, 15.4 million dollars
_MONEY_RE = re.compile(
    r"(?:"
    r"(?P<sym1>" + _CURRENCY_SYMBOLS + r")\s*(?P<num1>" + _NUMBER + r")(?P<mag1>" + _MAGNITUDE + r")?"
    r"|"
    r"(?P<num2>" + _NUMBER + r")(?P<mag2>" + _MAGNITUDE + r")\s+(?P<sym2>dollars?|euros?|pounds?)"
    r")",
    re.IGNORECASE,
)

_MAGNITUDE_MAP = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillions": 1_000_000_000_000,
}

_ORG_STOPLIST = {"company", "llc", "sec", "ferc", "fasb", "gaap", "pcaob", "coso", "asu", "aicpa",
                 "diluted", "basic", "weighted", "equity", "controller", "treasury"}
_STOPLIST_NORMALIZED = {item.lower() for item in _ORG_STOPLIST}

# Multi-token boilerplate phrases that spaCy mis-tags as ORG in SEC filings.
# Normalized (lowercase, punctuation→space) for fast lookup.
_ORG_PHRASE_BLOCKLIST = {
    # Accounting document names
    "financial statements",
    "consolidated financial statements",
    "the consolidated financial statements",
    "consolidated balance sheets",
    "supplementary data",
    "financial statement schedules",
    # SEC filing boilerplate roles / generic terms
    "registrant",
    "registrants",
    "commission",
    "board",
    "board of directors",
    "the board of directors",
    "trustee",
    "general counsel",
    "sec file no",
    # Standards-body long-form names
    "committee of sponsoring organizations of the treadway commission",
    "the committee of sponsoring organizations of the treadway commission",
    "public company accounting oversight board",
    "the public company accounting oversight board",
    "accounting oversight board",
    # Section-heading phrases
    "management s discussion and analysis of financial condition",
    "management s discussion and analysis",
    # SEC exhibit / section titles mis-tagged as ORG
    "issuer purchases of equity",
    "issuer purchases of",
    "interactive data file",
    "financial data",
    "selected financial data",
    "management s report",
    "management s annual report",
    "security ownership of certain beneficial owners and management",
    "regulation s-k",
    "regulation s k",
    # Financial statement line items / section references mis-tagged as ORG
    "registrant s common equity",
    "internal control integrated framework",
    "internal control",
    "equity",
    "controller",
}

# Structural heading pattern: "Item 7", "Exhibit 21", etc.
_HEADING_LINE_RE = re.compile(r"(?i)^\s*(?:item|exhibit)\s+\d")

# Catches legal-act references ("Securities Exchange Act of 1934") and other
# structural false positives that don't reduce to a phrase-blocklist entry.
_ORG_BOILERPLATE_RE = re.compile(
    r"(?i)(?:"
    r"\bact\s+of\s+\d{4}\b"                    # "Exchange Act of 1934"
    r"|^(?:the\s+)?securities\s+(?:exchange\s+)?act\b"  # "Securities Exchange Act"
    r"|\bsarbanes.oxley\b"                      # "Sarbanes-Oxley Act"
    r"|\bfinancial\s+statements?\b"             # bare "Financial Statement(s)"
    r"|\bdiscussion\s+and\s+analysis\b"         # section-heading fragment
    r"|\bpurchases\s+of\s+equity\b"             # "Issuer Purchases of Equity"
    r"|\bregulation\s+s-?k\b"                   # "Regulation S-K"
    r")"
)

# Tokens that, when found immediately before a PERSON entity, confirm it's human
_PERSON_TITLES = {
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "prof", "prof.",
    "sir", "sen", "rep", "hon",
    "ceo", "cfo", "coo", "cto", "cso", "evp", "svp", "vp",
    "president", "chairman", "chairwoman", "director", "officer",
    "secretary", "treasurer", "general", "manager", "principal",
}

# Terms that appear as PERSON false-positives in SEC filings
_PERSON_BLOCKLIST = {
    "subsidiary", "subsidiaries", "product", "products", "division",
    "business", "operations", "revenue", "income", "expense", "expenses",
    "goodwill", "equity", "asset", "assets", "liability", "liabilities",
    "shares", "stock", "statement", "statements", "accounting", "fiscal",
    "lease", "leases", "item", "exhibit", "bermuda", "bermudas",
    # Financial/accounting terms mis-tagged as PERSON
    "earnings", "retained", "diluted", "weighted", "proceeds",
    "transactions", "regulatory", "depreciation", "amortization",
    # Governance / section-heading fragments
    "corporate", "governance", "compensation", "compliance",
    # US state names that appear in utility company names (e.g. "Ameren Missouri")
    "missouri", "illinois", "virginia", "carolina", "indiana", "ohio",
    "michigan", "wisconsin", "georgia", "florida", "texas", "nevada",
    # Technology/product terms mis-tagged as PERSON
    "cloud", "platform", "service", "services", "palantir", "software",
    "solution", "solutions", "network", "networks", "system", "systems",
}

_WORD_CURRENCY_MAP = {
    "dollar": "USD", "dollars": "USD",
    "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP",
}

_SYMBOL_CURRENCY_MAP = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
    "chf": "CHF", "usd": "USD", "eur": "EUR", "gbp": "GBP", "jpy": "JPY",
}


def _parse_number(num_str: str) -> Optional[float]:
    try:
        return float(num_str.replace(",", ""))
    except ValueError:
        return None


def _extract_monetary_values(text: str) -> list[dict]:
    results = []
    for m in _MONEY_RE.finditer(text):
        raw = m.group(0)
        if m.group("sym1"):
            sym = m.group("sym1").lower()
            currency = _SYMBOL_CURRENCY_MAP.get(sym, sym.upper())
            num_str = m.group("num1")
            mag_str = (m.group("mag1") or "").strip().lower()
        else:
            num_str = m.group("num2")
            mag_str = (m.group("mag2") or "").strip().lower()
            word = (m.group("sym2") or "").lower()
            currency = _WORD_CURRENCY_MAP.get(word, "USD")

        amount = _parse_number(num_str)
        if amount is not None and mag_str:
            multiplier = _MAGNITUDE_MAP.get(mag_str, 1)
            amount = amount * multiplier

        results.append({
            "text": raw.strip(),
            "entity_type": "monetary_value",
            "normalized_value": raw.strip(),
            "start_char": m.start(),
            "end_char": m.end(),
            "source": "regex",
            "currency": currency,
            "amount": amount,
            "raw_text": raw.strip(),
        })
    return results


# ── spaCy NER ─────────────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        from src.constants import SPACY_MODEL
        try:
            _nlp = spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer"])
        except OSError:
            logger.warning("spaCy model %s not found; NER disabled", SPACY_MODEL)
            _nlp = None
    return _nlp


def _extract_spacy_entities(text: str) -> list[dict]:
    nlp = _get_nlp()
    if nlp is None:
        return []

    # Truncate to avoid hitting spaCy's max_length on very large chunks
    truncated = text[:100_000]
    doc = nlp(truncated)

    results = []
    for ent in doc.ents:
        if ent.label_ not in ("ORG", "PERSON"):
            continue
        entity_type = "company" if ent.label_ == "ORG" else "person"
        if entity_type == "company" and not _keep_org_entity(ent.text):
            continue
        if entity_type == "person" and not _keep_person_entity(doc, ent):
            continue
        results.append({
            "text": ent.text,
            "entity_type": entity_type,
            "normalized_value": ent.text.strip(),
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "source": "spacy",
        })
    return results


def _keep_org_entity(text: str) -> bool:
    """Return False for obvious generic ORG false positives in SEC filings.

    Four-layer check (fast → slow):
    1. Single-token stoplist (e.g. "Company", "FASB", "COSO")
    2. Multi-word phrase blocklist (e.g. "Board of Directors", "Financial Statements")
    3. Regex patterns for structural false positives (legal-act refs, section headings)
    """
    normalized = re.sub(r"[^\w]+", " ", text).strip().lower()
    if not normalized:
        return False

    # Layer 1: single-token stoplist
    if normalized in _STOPLIST_NORMALIZED:
        return False
    tokens = normalized.split()
    if len(tokens) == 1 and tokens[0] in _STOPLIST_NORMALIZED:
        return False

    # Layer 2: multi-word phrase blocklist
    if normalized in _ORG_PHRASE_BLOCKLIST:
        return False

    # Layer 3: regex structural patterns
    if _ORG_BOILERPLATE_RE.search(text):
        return False

    return True


def _person_entity_line(doc_text: str, char_pos: int) -> str:
    """Return the source line containing char_pos."""
    start = doc_text.rfind("\n", 0, char_pos)
    start = 0 if start == -1 else start + 1
    end = doc_text.find("\n", char_pos)
    return doc_text[start:] if end == -1 else doc_text[start:end]


def _line_is_table_row(line: str) -> bool:
    """True when the line looks like a space-aligned or pipe-separated table row."""
    if "|" in line:
        return True
    # Three or more standalone numbers on one line → numeric table row
    return len(re.findall(r"(?<!\w)\$?[\d,]+(?:\.\d+)?(?!\w)", line)) >= 3


def _keep_person_entity(doc, ent) -> bool:
    """Filter PERSON entities using spaCy token features and context.

    Layer 1: fast string filters (blocklist, digits, capitalisation, alpha)
    Layer 2: line-context filters (table rows, structural headings)
    Layer 3: single-token gate (must be preceded by a title token)
    Layer 4: POS tags (all PROPN)
    Layer 5: title-context override
    """
    tokens = [t for t in ent]

    # Layer 1: string filters
    normalized_tokens = [re.sub(r"[^\w]+", "", t.text).lower() for t in tokens]
    if not any(normalized_tokens):
        return False
    if any(tok in _PERSON_BLOCKLIST for tok in normalized_tokens):
        return False
    # Reject any token that contains a digit character
    if any(c.isdigit() for t in tokens for c in t.text):
        return False
    # At least one token must start with an uppercase letter
    if not any(t.text[:1].isupper() for t in tokens):
        return False
    # At least one token must contain an alphabetic character
    if not any(any(c.isalpha() for c in t.text) for t in tokens):
        return False

    # Layer 2: line-context filters
    line = _person_entity_line(doc.text, ent.start_char)
    if _line_is_table_row(line):
        return False
    if _HEADING_LINE_RE.match(line):
        return False

    if len(tokens) == 1:
        # Only keep single-token PER if preceded by a title (e.g. "Dr. Smith")
        prev_idx = ent.start - 1
        if prev_idx < 0:
            return False
        prev = doc[prev_idx].text.lower().rstrip(".")
        return prev in _PERSON_TITLES

    # Layer 4: spaCy POS — all tokens should be proper nouns
    all_propn = all(t.pos_ == "PROPN" for t in tokens)
    if all_propn:
        return True

    # Layer 5: title context overrides POS uncertainty
    prev_idx = ent.start - 1
    if prev_idx >= 0:
        prev = doc[prev_idx].text.lower().rstrip(".")
        if prev in _PERSON_TITLES:
            return True

    return False


# ── public interface ──────────────────────────────────────────────────────────

def extract_entities(chunk: dict) -> list[dict]:
    """Extract companies, people, and monetary values from chunk['section_text'].

    Returns a flat list of entity mention dicts.
    """
    text = chunk.get("section_text", "")
    if not text:
        return []

    entities = []
    entities.extend(_extract_spacy_entities(text))
    entities.extend(_extract_monetary_values(text))
    return entities
