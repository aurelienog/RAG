from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_DATASETS: Path = DATA_DIR / "datasets"
DATA_OUTPUT: Path = DATA_DIR / "output"

SEARCH_RESULTS_DIR: Path = DATA_OUTPUT / "search_results"
SEARCH_RESULTS_AND_ANSWER_DIR: Path = DATA_OUTPUT / "search_results_and_answer"

DEFAULT_MAX_CHUNK_SIZE: int = 2000

ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
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
