"""General-purpose helpers: logging, serialization, file I/O."""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def setup_logging(level: int = logging.INFO, filename: Path = None) -> logging.Logger:
    import time
    from src.constants import LOG_DATE_FORMAT, LOGS_DIR, LOG_FORMAT

    if filename is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = LOGS_DIR / f"pipeline_{stamp}.log"
    filename.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        root.addHandler(stream)

        fh = logging.FileHandler(filename, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger("pipeline")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def make_section_id(source_path: str, section_heading: str) -> str:
    raw = f"{source_path}|{section_heading}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def make_document_id(source_path: str) -> str:
    return hashlib.md5(source_path.encode()).hexdigest()[:12]


def _serialize_value(v: Any) -> Any:
    """Recursively convert non-JSON-native types."""
    if isinstance(v, dict):
        return {k: _serialize_value(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    if isinstance(v, float) and (v != v):  # NaN
        return None
    return v


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(_serialize_value(obj), ensure_ascii=False)


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(safe_json_dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_parquet(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize nested columns to JSON strings so pyarrow can handle them uniformly.
    nested_cols = {"entities", "resolved_entities", "tables", "page_numbers", "processing_warnings"}
    flat_records = []
    for r in records:
        row = dict(r)
        for col in nested_cols:
            if col in row:
                row[col] = safe_json_dumps(row[col])
        flat_records.append(row)
    df = pd.DataFrame(flat_records)
    df.to_parquet(path, index=False)


def truncate_text(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
