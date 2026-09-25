from __future__ import annotations

from pathlib import Path


ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT / "data"

DATA_RAW: Path = DATA_DIR / "raw" / "vllm-0.10.1"
DATA_PROCESSED: Path = DATA_DIR / "processed"

DATA_DATASETS: Path = DATA_DIR / "datasets"
DATA_OUTPUT: Path = DATA_DIR / "output"

SEARCH_RESULTS_DIR: Path = DATA_OUTPUT / "search_results"
SEARCH_RESULTS_AND_ANSWER_DIR: Path = (
    DATA_OUTPUT / "search_results_and_answer"
)

DEFAULT_MAX_CHUNK_SIZE: int = 2000


# Public retrieval dataset.
UNANSWERED_DOCS_DATASET: Path = (
    DATA_DATASETS
    / "UnansweredQuestions"
    / "dataset_docs_public.json"
)

# Ground-truth dataset used by the local evaluate command.
ANSWERED_DOCS_DATASET: Path = (
    DATA_DATASETS
    / "AnsweredQuestions"
    / "dataset_docs_public.json"
)

# Default output location for search_dataset.
SEARCH_RESULTS_UNANSWERED_DOCS_DIR: Path = (
    SEARCH_RESULTS_DIR / "UnansweredQuestions"
)

# Default student search-results file used by evaluate.
RESULTS_PATH: Path = (
    SEARCH_RESULTS_UNANSWERED_DOCS_DIR
    / "dataset_docs_public.json"
)


ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cu",
}


IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    ".tox",
}
