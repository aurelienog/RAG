from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_DATASETS = DATA_DIR / "datasets"
DATA_OUTPUT = DATA_DIR / "output"

SEARCH_RESULTS_DIR = DATA_OUTPUT / "search_results"
SEARCH_RESULTS_AND_ANSWER_DIR = DATA_OUTPUT / "search_results_and_answer"

DEFAULT_MAX_CHUNK_SIZE = 2000
