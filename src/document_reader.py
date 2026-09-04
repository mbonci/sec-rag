"""Load documents into a normalized representation.

Preserves the notebook's helper functions for .txt loading.
Optional PDF support via PyMuPDF with quality-based routing.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from src.constants import (
    DATA_DIR,
    PDF_MAX_REPLACEMENT_RATIO,
    PDF_MIN_ALPHA_RATIO,
    PDF_MIN_TEXT_LENGTH,
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_TXT_EXTENSIONS,
)
from src.utils import make_document_id

logger = logging.getLogger(__name__)


# ── txt helpers (preserved from notebook) ────────────────────────────────────

def get_txt_files_from_directory(directory_path: str) -> list[str]:
    """Return .txt file paths in directory_path."""
    return [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".txt")
    ]


def read_txt_files(directory_path: str) -> list[dict]:
    """Read all .txt files; return list of dicts with 'text' and 'filepath'."""
    documents = []
    for file_path in get_txt_files_from_directory(directory_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            documents.append({"text": f.read(), "filepath": file_path})
    return documents


# ── normalized document representation ───────────────────────────────────────

def load_documents(
    directory: Path = DATA_DIR,
    limit: Optional[int] = None,
    include_pdfs: bool = False,
) -> list[dict]:
    """Load all supported documents from directory into normalized dicts.

    Returns list of:
      {document_id, source_path, file_type, raw_text, metadata}
    """
    docs = []
    paths = sorted(directory.iterdir())
    if limit:
        # Apply limit across combined file types
        paths = paths[:limit * 2]  # over-select then trim after filtering

    for p in paths:
        ext = p.suffix.lower()
        if ext in SUPPORTED_TXT_EXTENSIONS:
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
                docs.append(_make_doc(p, "txt", raw))
            except Exception as e:
                logger.warning("Failed to read %s: %s", p.name, e)
        elif include_pdfs and ext in SUPPORTED_PDF_EXTENSIONS:
            doc = _load_pdf(p)
            if doc:
                docs.append(doc)

        if limit and len(docs) >= limit:
            break

    logger.info("Loaded %d documents from %s", len(docs), directory)
    return docs


def _make_doc(path: Path, file_type: str, raw_text: str) -> dict:
    return {
        "document_id": make_document_id(str(path)),
        "source_path": str(path),
        "file_type": file_type,
        "raw_text": raw_text,
        "metadata": {"filename": path.name},
    }


# ── optional PDF support ──────────────────────────────────────────────────────

def _load_pdf(path: Path) -> Optional[dict]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping PDF %s", path.name)
        return None

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.warning("Cannot open PDF %s: %s", path.name, e)
        return None

    pages = []
    full_text_parts = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        quality = assess_pdf_page_quality(text)
        page_rec = {
            "page_number": page_num,
            "text": text,
            "extraction_method": "pymupdf",
            "quality_score": quality["score"],
            "warnings": quality["warnings"],
        }
        pages.append(page_rec)
        full_text_parts.append(text)

    raw_text = "\n".join(full_text_parts)
    result = _make_doc(path, "pdf", raw_text)
    result["metadata"]["pages"] = pages
    result["metadata"]["page_count"] = len(pages)
    return result


def assess_pdf_page_quality(text: str) -> dict:
    """Return a quality dict for a single PDF page's extracted text."""
    warnings = []
    n = len(text)

    if n < PDF_MIN_TEXT_LENGTH:
        warnings.append("very_short_text")

    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / max(n, 1)
    if alpha_ratio < PDF_MIN_ALPHA_RATIO and n >= PDF_MIN_TEXT_LENGTH:
        warnings.append("low_alpha_ratio")

    replacement_chars = text.count("�")
    repl_ratio = replacement_chars / max(n, 1)
    if repl_ratio > PDF_MAX_REPLACEMENT_RATIO:
        warnings.append("high_replacement_chars")

    # Highly fragmented: average line length < 10 chars
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        avg_line = sum(len(l) for l in lines) / len(lines)
        if avg_line < 10:
            warnings.append("fragmented_lines")

    score = max(0.0, 1.0 - 0.25 * len(warnings))
    return {
        "score": round(score, 2),
        "warnings": warnings,
        "requires_fallback": len(warnings) >= 2 or "very_short_text" in warnings,
    }
