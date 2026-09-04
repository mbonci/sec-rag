"""Main orchestration: load → parse → extract → resolve → tables → save → standardize."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src import constants
import asyncio

from src.document_reader import load_documents
from src.entity_extractor import extract_entities
from src.entity_resolver import attach_resolved_entities, resolve_entities
from src.section_parser import parse_sections
from src.table_parser import detect_and_parse_tables, parse_tables_batch
from src.utils import setup_logging, write_jsonl

logger = logging.getLogger(__name__)


def run_pipeline(
    input_dir: Path = constants.DATA_DIR,
    output_jsonl: Path = constants.OUTPUT_JSONL,
    output_standardized: Optional[Path] = None,
    limit: Optional[int] = None,
    include_pdfs: bool = False,
    use_llm_table_summaries: bool = False,
    stop_after: int = 7,
    verbose: bool = False,
) -> list[dict]:
    """Run the full document processing pipeline.

    Returns the list of chunk records (also saved to disk).
    """
    setup_logging(logging.DEBUG if verbose else logging.INFO)

    # Stage 1: Load documents
    logger.info("Stage 1: Loading documents from %s", input_dir)
    documents = load_documents(input_dir, limit=limit, include_pdfs=include_pdfs)
    if not documents:
        logger.error("No documents loaded. Check input_dir: %s", input_dir)
        return []

    # Stage 2: Parse sections
    logger.info("Stage 2: Parsing sections (%d documents)", len(documents))
    all_sections: list[dict] = []
    for doc in documents:
        try:
            sections = parse_sections(doc)
            all_sections.extend(sections)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", doc.get("source_path"), e)

    logger.info("Produced %d sections", len(all_sections))

    # Stage 3: Entity extraction
    logger.info("Stage 3: Extracting entities")
    for section in all_sections:
        try:
            section["entities"] = extract_entities(section)
        except Exception as e:
            logger.warning("Entity extraction failed on section %s: %s", section.get("section_heading"), e)
            section["processing_warnings"].append(f"entity_extraction_error: {e}")

    # Stage 4: Entity resolution (per document to prevent cross-document false merges)
    logger.info("Stage 4: Resolving entities")
    doc_sections: dict[str, list[dict]] = defaultdict(list)
    for section in all_sections:
        doc_sections[section["document_id"]].append(section)

    total_groups = 0
    for sections in doc_sections.values():
        resolution_map = resolve_entities(sections)
        attach_resolved_entities(sections, resolution_map)
        total_groups += len(resolution_map)
    logger.info("Resolved %d entity groups across %d documents", total_groups, len(doc_sections))

    if stop_after <= 4:
        logger.info("Stopping after Stage 4 as requested — saving partial output.")
        write_jsonl(all_sections, output_jsonl)
        logger.info("Done. Output: %s", output_jsonl)
        return all_sections

    # Stage 5: Table detection and parsing (concurrent LLM calls via parse_tables_batch)
    logger.info("Stage 5: Detecting and parsing tables")
    try:
        asyncio.run(parse_tables_batch(all_sections, use_llm_summaries=use_llm_table_summaries))
    except Exception as e:
        logger.warning("Batch table parsing failed (%s) — falling back to sequential", e)
        for section in all_sections:
            try:
                section["tables"] = detect_and_parse_tables(
                    section, use_llm_summaries=use_llm_table_summaries
                )
            except Exception as e2:
                logger.warning("Table parsing failed on section %s: %s", section.get("section_heading"), e2)
                section["processing_warnings"].append(f"table_parsing_error: {e2}")

    sections_with_tables = sum(1 for s in all_sections if s["tables"])
    logger.info("Found tables in %d sections", sections_with_tables)

    # Stage 6: Save raw outputs
    logger.info("Stage 6: Saving outputs")
    write_jsonl(all_sections, output_jsonl)

    # Stage 7: Standardize for RAG
    if output_standardized is not None:
        logger.info("Stage 7: Standardizing output → %s", output_standardized)
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
            from standardize_output import standardize_records
            standardize_records(all_sections, output_standardized)
        except Exception as e:
            logger.warning("Standardization failed: %s", e)

    logger.info("Done. Output: %s", output_jsonl)
    return all_sections
