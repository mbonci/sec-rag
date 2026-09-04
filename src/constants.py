"""Central configuration for paths, thresholds, and labels."""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

REPO_ROOT = Path(__file__).parent.parent

# Redirect HuggingFace model cache into the repo so models stay with the project.
# Must be set before any transformers / sentence-transformers import.
HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

# PyTorch 2.x on macOS dispatches some generation ops to MPS even when the
# model is on CPU, causing a segfault. Force CPU-only inference.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["PYTORCH_MPS_PREFER_METAL"] = "0"

DATA_DIR = REPO_ROOT / "data" / "Documents"
OUTPUTS_DIR = REPO_ROOT / "outputs"
INTERMEDIATE_DIR = OUTPUTS_DIR / "intermediate"
FINAL_DIR = OUTPUTS_DIR / "final"
CACHE_DIR = OUTPUTS_DIR / "cache"
COSTS_DIR = OUTPUTS_DIR / "costs"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
LOGS_DIR = REPO_ROOT / "logs"

OUTPUT_JSONL = FINAL_DIR / "chunks.jsonl"
OUTPUT_PARQUET = FINAL_DIR / "chunks.parquet"
EVALUATION_CSV = EVALUATION_DIR / "eval_sample.csv"
SUMMARY_JSON = EVALUATION_DIR / "summary.json"
LOG_FILE = LOGS_DIR / "pipeline.log"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

SUPPORTED_TXT_EXTENSIONS = {".txt"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

SPACY_MODEL = "en_core_web_sm"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Chunking
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200

# Entity resolution
ENTITY_SIMILARITY_THRESHOLD = 85  # RapidFuzz score 0–100

# Table parsing
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

# PDF quality thresholds
PDF_MIN_ALPHA_RATIO = 0.4       # flag pages below this alphabetic-char ratio
PDF_MIN_TEXT_LENGTH = 50        # flag pages with fewer chars than this
PDF_MAX_REPLACEMENT_RATIO = 0.1 # flag pages where >10% chars are replacement chars

# Evaluation sample
EVAL_SAMPLE_SIZE = 20
RANDOM_SEED = 42

# Document sample limit (None = all)
DEFAULT_DOCUMENT_LIMIT = None

# Section taxonomy
SECTION_CATEGORIES = [
    "business_overview",
    "risk_factors",
    "financial_results",
    "management_discussion",
    "legal_proceedings",
    "governance",
    "notes_to_financial_statements",
    "properties",
    "disclosures",
    "market_information",
    "exhibits",
    "other",
]

# Item-number → category map (SEC 10-K structure is fixed by regulation)
# Used as the primary category lookup; title-text rules are the fallback.
ITEM_CATEGORY_MAP: dict[str, str] = {
    "1":   "business_overview",
    "1A":  "risk_factors",
    "1B":  "disclosures",      # Unresolved Staff Comments
    "2":   "properties",       # Properties
    "3":   "legal_proceedings",
    "4":   "disclosures",      # Mine Safety Disclosures
    "5":   "market_information",  # Market for Registrant's Common Equity
    "6":   "financial_results",
    "7":   "management_discussion",
    "7A":  "management_discussion",
    "8":   "financial_results",
    "9":   "financial_results",
    "9A":  "disclosures",      # Controls and Procedures
    "9B":  "disclosures",      # Other Information
    "9C":  "disclosures",      # Disclosure Regarding Foreign Jurisdictions
    "10":  "governance",
    "11":  "governance",
    "12":  "governance",
    "13":  "governance",
    "14":  "governance",
    "15":  "exhibits",         # Exhibits and Financial Statement Schedules
    "16":  "exhibits",         # Form 10-K Summary
}

# Heading → category mapping rules (checked in order; first match wins)
# Each entry: (regex_pattern, category)
HEADING_CATEGORY_RULES = [
    # ITEM 1. BUSINESS  (not ITEM 1A)
    (r"item\s+1[^a-z0-9].*business", "business_overview"),
    (r"business\s+overview", "business_overview"),
    (r"item\s+1a[^a-z0-9].*risk", "risk_factors"),
    (r"risk\s+factors?", "risk_factors"),
    (r"item\s+6\b.*financial\s+data", "financial_results"),
    (r"item\s+8\b.*financial\s+statements", "financial_results"),
    (r"selected\s+financial\s+data", "financial_results"),
    (r"financial\s+statements?", "financial_results"),
    (r"results?\s+of\s+operations?", "financial_results"),
    (r"item\s+7[^a]\b.*management", "management_discussion"),
    (r"management.{0,10}discussion", "management_discussion"),
    (r"md&a", "management_discussion"),
    (r"item\s+3\b.*legal", "legal_proceedings"),
    (r"legal\s+proceedings?", "legal_proceedings"),
    (r"item\s+10\b.*directors?", "governance"),
    (r"item\s+11\b.*compensation", "governance"),
    (r"item\s+12\b.*security\s+ownership", "governance"),
    (r"item\s+13\b.*related\s+transactions?", "governance"),
    (r"item\s+14\b.*accounting\s+fees", "governance"),
    (r"corporate\s+governance", "governance"),
    (r"executive\s+compensation", "governance"),
    (r"note\s+\d+", "notes_to_financial_statements"),
    (r"notes?\s+to\s+.*financial\s+statements?", "notes_to_financial_statements"),
    # Properties
    (r"item\s+2\b.*propert", "properties"),
    (r"^propert(?:y|ies)\b", "properties"),
    # Market information / common equity
    (r"item\s+5\b.*(?:market|equity|stock)", "market_information"),
    (r"market\s+for\s+.*common\s+(?:equity|stock)", "market_information"),
    (r"common\s+(?:equity|stock).*market\s+(?:price|information)", "market_information"),
    (r"(?:stock|share)\s+(?:price|performance|repurchase)", "market_information"),
    # Exhibits and financial schedules
    (r"item\s+15\b.*exhibit", "exhibits"),
    (r"item\s+16\b", "exhibits"),
    (r"^exhibits?\b", "exhibits"),
    (r"financial\s+statement\s+schedules?", "exhibits"),
    # Disclosures (unresolved staff comments, controls, other info, mine safety)
    (r"item\s+1b\b.*unresolved", "disclosures"),
    (r"unresolved\s+staff\s+comments?", "disclosures"),
    (r"item\s+4\b.*mine\s+safety", "disclosures"),
    (r"mine\s+safety\s+disclosures?", "disclosures"),
    (r"item\s+9a\b.*controls?", "disclosures"),
    (r"controls?\s+and\s+procedures?", "disclosures"),
    (r"item\s+9b\b.*other\s+information", "disclosures"),
    (r"item\s+9c\b", "disclosures"),
    (r"disclosure\s+regarding\s+foreign\s+jurisdictions?", "disclosures"),
]
