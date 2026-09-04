# SEC 10-K RAG Pipeline

A document processing pipeline over SEC 10-K filings that produces a structured, embedding-ready dataset for retrieval-augmented generation. Built as a way to get hands-on with the full RAG stack — from raw HTML filings to a local REPL that answers questions over a corpus of financial documents.

---

## What it does

Processes 50 SEC 10-K filings through a modular pipeline:

| Stage | Description |
|---|---|
| 1 · Document loading | Reads `.txt` (and optional `.pdf`) files; assigns stable `document_id` |
| 2 · Section parsing | Detects Item headings via TOC extraction (LLM-assisted) + regex fallback; assigns one of 12 section categories |
| 3 · Entity extraction | spaCy NER for `company` / `person`; regex for `monetary_value` |
| 4 · Entity resolution | RapidFuzz fuzzy matching groups variant mentions to a canonical name per document |
| 5 · Table parsing | Detects space-aligned and pipe-separated tables; infers column headers; generates LLM summaries (cached) |
| 6 · Save JSONL | Writes `outputs/final/chunks.jsonl` |
| 7 · Standardize | Flattens to a parquet schema ready for embedding and retrieval |

Output: `outputs/final/chunks.parquet` — 1,069 chunks across 50 documents, 26 columns.

The processed corpus feeds a local RAG REPL (`app.py`) using FAISS + a quantized Qwen model via Apple MLX.

---

## Quick Start

```bash
# 1. Create environment (Python 3.12)
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Set your OpenAI key (used for TOC extraction + table summaries, cached after first run)
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Unzip the documents
cd data && unzip Documents.zip && cd ..

# 5. Run the full pipeline
python scripts/run_pipeline.py --standardize --use-llm-table-summaries

# 6. Validate results
python scripts/validate_outputs.py --input outputs/final/chunks.jsonl

# 7. Run the RAG demo
python app.py
```

Faster options:

```bash
# Skip table parsing (much faster)
python scripts/run_pipeline.py --stop-after 4

# Smoke test with 5 documents
python scripts/run_pipeline.py --limit 5

# Rebuild FAISS index after re-running
python scripts/build_rag_index.py

# Run tests
python -m pytest tests/ -v
```

---

## Code Structure

```
src/
  constants.py          # paths, thresholds, category taxonomy
  document_reader.py    # .txt / PDF loader
  section_parser.py     # heading detection, TOC extraction, chunking
  entity_extractor.py   # spaCy NER + monetary regex
  entity_resolver.py    # RapidFuzz resolution
  table_parser.py       # table detection, column inference, LLM summaries
  pipeline.py           # orchestration (Stages 1-7)
  retrieval.py          # FAISS search helper
  generation.py         # MLX local LLM generation
  rag_prompt.py         # system prompt and context formatter

scripts/
  run_pipeline.py       # CLI entry point
  build_rag_index.py    # embed chunks -> FAISS index
  standardize_output.py # JSONL -> parquet
  validate_outputs.py   # validation report + PNG plot
  inspect_outputs.py    # HTML chunk inspector

tests/                  # 47 unit tests
app.py                  # interactive RAG REPL
```

---

## RAG Demo

```
python app.py
> What were Exelon's total revenues in 2018?
> What risks did Dominion Energy disclose related to climate change?
```

- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, sentence-transformers)
- **Index:** FAISS `IndexFlatIP` with cosine similarity
- **Generation:** `Qwen2.5-1.5B-Instruct-4bit` via Apple MLX (no PyTorch, runs on Apple Silicon)

Example outputs are in [`outputs/evaluation/llm_rag_examples/`](outputs/evaluation/llm_rag_examples/).

---

## Section Categories

`business_overview` · `risk_factors` · `financial_results` · `management_discussion` · `legal_proceedings` · `governance` · `notes_to_financial_statements` · `properties` · `disclosures` · `market_information` · `exhibits` · `other`

Mapped from SEC 10-K Item numbers, which are fixed by regulation.

---

## LLM Usage

LLM calls (OpenAI `gpt-4.1-mini`) are used in two places, both cached by content hash:

| Use | Purpose |
|---|---|
| TOC extraction | Parse table-of-contents to locate section boundaries (only when regex fails) |
| Table summaries | 1–3 sentence description of what each table shows |

Total cost across all development runs: ~$0.95. Re-running from cache is free.
