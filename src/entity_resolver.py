"""Entity resolution: group variant mentions into canonical forms.

The resolver intentionally favors false negatives over false merges. It uses
simple, explainable rules: type-specific normalization, exact normalized
matching, conservative acronym matching, and strict fuzzy agreement.
"""

import logging
import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz

from src.constants import ENTITY_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]")
_POSSESSIVE_RE = re.compile(r"(?:'s|’s)\s*$", re.IGNORECASE)
_TITLE_RE = re.compile(r"^(?:mr|mrs|ms|dr|prof|sir)\.?\s+", re.IGNORECASE)

_LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "plc", "gmbh", "ag", "sa", "bv", "lp", "llp", "l", "p", "co",
}
_LEADING_ARTICLES = {"the", "an"}
_ACRONYM_IGNORED_TOKENS = {"and", "of", "the", "a", "an", "companies", "company"}
_GENERIC_CANONICAL_TOKENS = {
    "company", "corporation", "inc", "llc", "ltd", "consolidated", "financial",
    "statements", "statement", "exhibit", "item", "registrant", "trustee",
}
_MEANINGFUL_CONTAINMENT_TOKENS = {
    "group", "holdings", "partners", "capital", "university", "health", "media",
    "gas", "energy", "school", "business", "fund", "trust", "bank", "services",
    "technologies", "communications", "securities", "transaction", "agreement",
}
_PERSON_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _normalize(name: str) -> str:
    """Backward-compatible company normalizer used by existing tests."""
    return _normalize_company(name)


def _clean_text(name: str) -> str:
    name = _POSSESSIVE_RE.sub("", name.strip())
    name = name.replace("\xa0", " ")
    name = _PUNCT_RE.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def _strip_leading_articles(tokens: list[str]) -> list[str]:
    while len(tokens) > 1 and tokens[0] in _LEADING_ARTICLES:
        tokens = tokens[1:]
    return tokens


def _normalize_company(name: str) -> str:
    """Normalize company/org names without erasing meaningful business terms."""
    tokens = _strip_leading_articles(_clean_text(name).split())
    if len(tokens) > 1:
        while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIX_TOKENS:
            tokens = tokens[:-1]
    return " ".join(tokens)


def _normalize_person(name: str) -> str:
    """Normalize person names, keeping enough signal to avoid surname merges."""
    clean = _TITLE_RE.sub("", _clean_text(name))
    tokens = [t for t in clean.split() if t not in _PERSON_SUFFIXES]
    normalized = []
    for token in tokens:
        if len(token) == 1:
            normalized.append(token)
        elif len(token) == 2 and token.endswith("."):
            normalized.append(token[0])
        else:
            normalized.append(token)
    return " ".join(normalized)


def _is_likely_acronym_match(short: str, long: str) -> bool:
    """True when an all-caps short form is clear evidence for a longer name."""
    if len(short) < 2 or len(short) > 6 or not short.isupper():
        return False
    words = [w for w in long.split() if w not in _ACRONYM_IGNORED_TOKENS]
    if len(words) < 2:
        return False
    if words[0].upper() == short:
        return False

    initials = "".join(w[0].upper() for w in words if w)
    if short == initials:
        return True

    # Conservative known expansion for common SEC filing abbreviations:
    # Interpublic is commonly abbreviated as IP in IPG.
    if short == "IPG" and len(words) >= 2 and words[0] == "interpublic" and words[1] == "group":
        return True

    return False


def _fuzzy_scores(a: str, b: str) -> dict[str, float]:
    return {
        "ratio": fuzz.ratio(a, b),
        "token_sort": fuzz.token_sort_ratio(a, b),
        "token_set": fuzz.token_set_ratio(a, b),
        "partial": fuzz.partial_ratio(a, b),
    }


def _has_meaningful_containment(short_norm: str, long_norm: str) -> bool:
    short_tokens = set(short_norm.split())
    long_tokens = long_norm.split()
    if not short_tokens or not short_tokens.issubset(long_tokens):
        return False
    extra = set(long_tokens) - short_tokens
    return bool(extra & _MEANINGFUL_CONTAINMENT_TOKENS)


def _should_merge_company(a: str, b: str) -> tuple[bool, str, int]:
    na, nb = _normalize_company(a), _normalize_company(b)
    if not na or not nb:
        return False, "no_match", 0
    if na == nb:
        return True, "normalized_exact", 100

    if _has_meaningful_containment(na, nb) or _has_meaningful_containment(nb, na):
        return False, "no_match", 0

    if _is_likely_acronym_match(a.strip(), nb) or _is_likely_acronym_match(b.strip(), na):
        return True, "acronym", 100

    if min(len(na), len(nb)) < 5:
        return False, "no_match", 0

    scores = _fuzzy_scores(na, nb)
    required = max(90, ENTITY_SIMILARITY_THRESHOLD)
    direct_ok = scores["ratio"] >= required
    token_ok = scores["token_sort"] >= required and scores["token_set"] >= required
    if direct_ok and token_ok:
        return True, "strict_fuzzy", int(min(scores["ratio"], scores["token_sort"], scores["token_set"]))
    return False, "no_match", int(max(scores.values()))


def _person_tokens_without_initials(norm: str) -> list[str]:
    return [t for t in norm.split() if len(t) > 1]


def _entity_block_keys(name: str, entity_type: str) -> list[str]:
    """Return conservative blocking keys used to cut pairwise comparisons.

    Blocks are intentionally narrow because the resolver prefers false
    negatives over false merges.
    """
    normalizer = _normalize_company if entity_type == "company" else _normalize_person
    normalized = normalizer(name)
    if not normalized:
        return []

    tokens = normalized.split()
    if not tokens:
        return []

    blocks = [f"norm:{normalized}"]

    if entity_type == "company":
        blocks.append(f"first:{tokens[0]}")
        blocks.append(f"last:{tokens[-1]}")
        if len(tokens) >= 2:
            blocks.append(f"fl:{tokens[0]}:{tokens[-1]}")
        acronym = "".join(t[0] for t in tokens if t and t not in _ACRONYM_IGNORED_TOKENS)
        if 2 <= len(acronym) <= 6:
            blocks.append(f"acro:{acronym}")
    else:
        blocks.append(f"last:{tokens[-1]}")
        if len(tokens) >= 2:
            blocks.append(f"first:{tokens[0]}")
            blocks.append(f"fl:{tokens[0]}:{tokens[-1]}")
            blocks.append(f"fi:{tokens[0][0]}:{tokens[-1]}")

    return list(dict.fromkeys(blocks))


def _should_merge_person(a: str, b: str) -> tuple[bool, str, int]:
    na, nb = _normalize_person(a), _normalize_person(b)
    if not na or not nb:
        return False, "no_match", 0
    if na == nb:
        return True, "normalized_exact", 100

    ta, tb = na.split(), nb.split()
    if len(ta) < 2 or len(tb) < 2:
        return False, "no_match", 0

    long_a, long_b = _person_tokens_without_initials(na), _person_tokens_without_initials(nb)
    if len(long_a) >= 2 and len(long_b) >= 2 and long_a[0] == long_b[0] and long_a[-1] == long_b[-1]:
        return True, "person_name", 95
    if long_a and long_b and long_a[-1] == long_b[-1]:
        initials_a = "".join(t[0] for t in ta[:-1])
        initials_b = "".join(t[0] for t in tb[:-1])
        if initials_a and initials_b and initials_a == initials_b:
            return True, "person_initials", 95

    scores = _fuzzy_scores(na, nb)
    if scores["ratio"] >= 93 and scores["token_sort"] >= 93:
        return True, "strict_fuzzy", int(min(scores["ratio"], scores["token_sort"]))
    return False, "no_match", int(max(scores.values()))


def resolve_entities(chunks: list[dict]) -> list[dict]:
    """Group entity mentions across chunks into resolved canonical groups.

    Returns a list of resolution group dicts:
      {canonical_name, entity_type, variants, mention_count, resolution_method}

    Operates per entity_type independently.
    """
    # Collect all mentions by type
    by_type: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        for ent in chunk.get("entities", []):
            if ent.get("entity_type") in ("company", "person"):
                by_type[ent["entity_type"]].append(ent["text"])

    groups = []
    for entity_type, mentions in by_type.items():
        groups.extend(_resolve_mentions(mentions, entity_type))

    return groups


def _resolve_mentions(mentions: list[str], entity_type: str) -> list[dict]:
    """Cluster raw mentions into canonical resolution groups for one entity type.

    The routine works in three stages: first it merges exact normalized matches,
    then it compares only mentions that share a conservative blocking key, and
    finally it handles a small acronym edge case for companies. The design is
    intentionally conservative and prefers missed merges over incorrect ones.
    """

    if not mentions:
        return []

    counts = Counter(mentions)
    unique = list(counts.keys())

    parent = {m: m for m in unique}
    merge_evidence: dict[tuple[str, str], tuple[str, int]] = {}

    def find(x):
        """Return the current root for a mention, with path compression."""

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, method: str, score: int):
        """Merge two mentions and record the evidence for that pair."""

        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        merge_evidence[tuple(sorted((a, b)))] = (method, score)

    normalizer = _normalize_company if entity_type == "company" else _normalize_person
    should_merge = _should_merge_company if entity_type == "company" else _should_merge_person
    norm_map = {m: normalizer(m) for m in unique}

    # Pass 1: exact normalized match
    norm_to_members: dict[str, str] = {}
    for m in unique:
        n = norm_map[m]
        if not n:
            continue
        if n in norm_to_members:
            union(m, norm_to_members[n], "normalized_exact", 100)
        else:
            norm_to_members[n] = m

    # Pass 2: conservative type-specific merge rules, restricted to blocks.
    block_to_members: dict[str, list[str]] = defaultdict(list)
    for m in unique:
        for block_key in _entity_block_keys(m, entity_type):
            block_to_members[block_key].append(m)

    compared_pairs: set[tuple[str, str]] = set()
    for block_members in block_to_members.values():
        if len(block_members) < 2:
            continue
        for i in range(len(block_members)):
            left = block_members[i]
            for j in range(i + 1, len(block_members)):
                right = block_members[j]
                pair = tuple(sorted((left, right)))
                if pair in compared_pairs or find(left) == find(right):
                    continue
                compared_pairs.add(pair)
                merge, method, score = should_merge(left, right)
                if merge:
                    union(left, right, method, score)

    # Pass 2b: short all-caps strings are likely acronyms whose initials may not
    # reconstruct cleanly from the long form (e.g. "IPG" ← "Interpublic Group").
    # Blocking keys can't connect them, so compare them directly against all others.
    if entity_type == "company":
        caps_candidates = [m for m in unique if m.strip().isupper() and 2 <= len(m.strip()) <= 6]
        for caps in caps_candidates:
            for other in unique:
                if caps == other:
                    continue
                pair = tuple(sorted((caps, other)))
                if pair in compared_pairs or find(caps) == find(other):
                    continue
                compared_pairs.add(pair)
                merge, method, score = should_merge(caps, other)
                if merge:
                    union(caps, other, method, score)

    group_members: dict[str, list[str]] = defaultdict(list)
    for m in unique:
        group_members[find(m)].append(m)

    result = []
    for root, members in group_members.items():
        canonical = _pick_canonical(members, counts)
        mention_count = sum(counts[m] for m in members)
        variant_evidence = _build_variant_evidence(
            members, canonical, entity_type, merge_evidence, should_merge
        )
        methods = {
            e["resolution_method"]
            for e in variant_evidence
            if e["resolution_method"] != "canonical"
        }
        if not methods:
            methods = {"singleton"}
        method = methods.pop() if len(methods) == 1 else "mixed"
        result.append({
            "canonical_name": canonical,
            "entity_type": entity_type,
            "variants": sorted(set(members)),
            "mention_count": mention_count,
            "resolution_method": method,
            "variant_evidence": variant_evidence,
        })

    return sorted(result, key=lambda g: g["mention_count"], reverse=True)


def _build_variant_evidence(
    members: list[str],
    canonical: str,
    entity_type: str,
    merge_evidence: dict[tuple[str, str], tuple[str, int]],
    should_merge,
) -> list[dict]:
    evidence = []
    normalizer = _normalize_company if entity_type == "company" else _normalize_person
    for variant in sorted(set(members)):
        if variant == canonical:
            method, score = "canonical", 100
        elif normalizer(variant) == normalizer(canonical):
            method, score = "normalized_exact", 100
        else:
            key = tuple(sorted((variant, canonical)))
            if key in merge_evidence:
                method, score = merge_evidence[key]
            else:
                merge, method, score = should_merge(variant, canonical)
                if not merge:
                    method = "cluster_link"
        evidence.append({
            "variant": variant,
            "canonical_name": canonical,
            "resolution_method": method,
            "resolution_score": int(score),
        })
    return evidence


def _pick_canonical(members: list[str], counts: Counter) -> str:
    """Prefer frequent, formal, non-fragmentary, non-all-caps variants."""
    return max(members, key=lambda m: _canonical_score(m, counts))


def _canonical_score(name: str, counts: Counter) -> tuple[int, int, int, int, int]:
    clean = _clean_text(name)
    tokens = clean.split()
    generic_hits = sum(1 for token in tokens if token in _GENERIC_CANONICAL_TOKENS)
    has_legal_suffix = int(bool(tokens and tokens[-1] in _LEGAL_SUFFIX_TOKENS))
    non_upper = int(not name.isupper())
    multi_token = int(len(tokens) > 1)
    not_fragment = int(len(clean) >= 3 and "\n" not in name and not clean.isdigit())
    formality = has_legal_suffix + non_upper + multi_token + not_fragment - generic_hits
    return (formality, counts[name], non_upper, not_fragment, len(name))


def _evidence_for_variant(group: dict, variant: str) -> dict:
    for evidence in group.get("variant_evidence", []):
        if evidence.get("variant") == variant:
            return evidence
    return {
        "variant": variant,
        "canonical_name": group["canonical_name"],
        "resolution_method": group["resolution_method"],
        "resolution_score": None,
    }


def attach_resolved_entities(chunks: list[dict], resolution_map: list[dict]) -> list[dict]:
    """Add resolved_entities field to each chunk.

    Each chunk's resolved_entities is the subset of resolution groups that
    contain at least one entity mentioned in that chunk.
    """
    # Build lookup: variant → group
    variant_to_group: dict[str, dict] = {}
    for group in resolution_map:
        for v in group["variants"]:
            variant_to_group[v] = group

    for chunk in chunks:
        seen_canonicals: set[str] = set()
        resolved: list[dict] = []
        for ent in chunk.get("entities", []):
            text = ent.get("text", "")
            group = variant_to_group.get(text)
            if group and group["canonical_name"] not in seen_canonicals:
                seen_canonicals.add(group["canonical_name"])
                evidence = _evidence_for_variant(group, text)
                resolved.append({
                    "canonical_name": group["canonical_name"],
                    "entity_type": group["entity_type"],
                    "matched_text": text,
                    "resolution_method": evidence["resolution_method"],
                    "resolution_score": evidence["resolution_score"],
                })
        chunk["resolved_entities"] = resolved

    return chunks


def get_resolved_entities_for_chunk(chunk: dict, resolution_map: list[dict]) -> list[dict]:
    """Return resolved entities for a single chunk without mutating it.

    This is a convenience helper for notebook code that wants to do:
    chunk["resolved_entities"] = get_resolved_entities_for_chunk(chunk, resolution_map)
    """
    variant_to_group: dict[str, dict] = {}
    for group in resolution_map:
        for variant in group["variants"]:
            variant_to_group[variant] = group

    seen_canonicals: set[str] = set()
    resolved: list[dict] = []
    entities = chunk.get("entities", [])
    for ent in entities:
        text = ent.get("text", "")
        group = variant_to_group.get(text)
        if group and group["canonical_name"] not in seen_canonicals:
            seen_canonicals.add(group["canonical_name"])
            evidence = _evidence_for_variant(group, text)
            resolved.append({
                "canonical_name": group["canonical_name"],
                "entity_type": group["entity_type"],
                "matched_text": text,
                "resolution_method": evidence["resolution_method"],
                "resolution_score": evidence["resolution_score"],
            })

    return resolved
