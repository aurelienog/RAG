from tqdm import tqdm

import json
from collections import Counter
from pathlib import Path

from ..config import (
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_MAX_CHUNK_SIZE,
    IGNORED_DIRS,
    ALLOWED_SUFFIXES)
from ..domain import Chunk, IndexingError
from .chunking import PythonChunker, MarkdownChunker
from ..utils import Tokenizer


class Indexer:
    def __init__(
            self,
            raw_dir: str | Path = DATA_RAW,
            processed_dir: str | Path = DATA_PROCESSED,
    ) -> None:

        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def index(
            self,
            max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, int]:

        if max_chunk_size <= 0 or max_chunk_size > 2000:
            raise IndexingError("max_chunk_size must be between 1 and 2000 characters.")

        if not self.raw_dir.exists():
            raise IndexingError(f"Input directory not found: {self.raw_dir}")
        if not self.raw_dir.is_dir():
            raise IndexingError(f"Expected a directory: {self.raw_dir}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[Chunk] = []
        files_indexed = 0

        for file_path in tqdm(self._iter_source_files(), desc="Indexing files", unit="file"):
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise IndexingError(f"Could not read file: {file_path}") from exc

            relative_path = self._to_project_relative_path(file_path)
            chunker = self._select_chunker(relative_path, max_chunk_size=max_chunk_size)
            file_chunks = chunker.chunk_file(relative_path, content)

            if file_chunks:
                chunks.extend(file_chunks)
                files_indexed += 1

        inverted_index: dict[str, list[dict[str, int | str]]] = {}
        doc_freq: dict[str, int] = {}
        doc_lengths: dict[str, int] = {}

        for chunk in tqdm(chunks, desc="Tokenizing chunks", unit="chunk"):
            tokens = Tokenizer.tokenize(chunk.text)
            doc_lengths[chunk.id] = len(tokens)
            seen_terms: set[str] = set()
            for term, count in Counter(tokens).items():
                entry = {"chunk_id": chunk.id, "tf": count}
                inverted_index.setdefault(term, []).append(entry)
                if term not in seen_terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1
                    seen_terms.add(term)

        avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0.0

        payload = {
            "chunks": [
                {
                    "id": chunk.id,
                    "file_path": chunk.file_path,
                    "text": chunk.text,
                    "first_character_index": chunk.start,
                    "last_character_index": chunk.end,
                }
                for chunk in chunks
            ],
            "inverted_index": inverted_index,
            "doc_freq": doc_freq,
            "avg_doc_length": avg_doc_length,
        }
        output_path = self.processed_dir / "index.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        return {
            "files_indexed": files_indexed,
            "chunks": len(chunks),
        }

    def _iter_source_files(self) -> list[Path]:
        """Return all source files under the raw directory."""
        if not self.raw_dir.exists():
            return []

        files = []
        for path in sorted(self.raw_dir.rglob("*")):
            if not path.is_file():
                continue
            if any(part in IGNORED_DIRS for part in path.parts):
                continue

            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue

            if path.is_file():
                files.append(path)
        return files

    def _select_chunker(
        self,
        relative_path: str,
        max_chunk_size: int,
    ):
        """Choose the chunker according to the file type."""
        suffix = Path(relative_path).suffix.lower()

        if suffix == ".py":
            return PythonChunker(max_chunk_size=max_chunk_size)
        return MarkdownChunker(max_chunk_size=max_chunk_size)

    @staticmethod
    def _to_project_relative_path(file_path: Path) -> str:
        """Retorna la ruta del archivo desde la raíz del proyecto (ej: data/raw/vllm-0.10.1/...)"""
        # Si ejecutas desde la raíz del repositorio, Path.cwd() es la raíz del proyecto.
        # file_path.relative_to(Path.cwd()) asegura que empiece por 'data/raw/...'
        try:
            return file_path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            # En caso de fallback, buscar dónde empieza 'data'
            parts = file_path.parts
            if "data" in parts:
                idx = parts.index("data")
                return Path(*parts[idx:]).as_posix()
            return file_path.as_posix()
