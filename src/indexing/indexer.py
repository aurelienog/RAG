from pathlib import Path

from tqdm import tqdm

from ..config import (
    ALLOWED_SUFFIXES,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_MAX_CHUNK_SIZE,
    IGNORED_DIRS,
    ROOT,
)
from ..domain import Chunk, IndexingError
from .chunking import MarkdownChunker, PythonChunker
from .lexical_index import LexicalIndexer
from .storage import IndexStorage


class Indexer:
    def __init__(
        self,
        raw_dir: str | Path = DATA_RAW,
        processed_dir: str | Path = DATA_PROCESSED,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.storage = IndexStorage(processed_dir)
        self.lexical_indexer = LexicalIndexer()

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, int]:

        if max_chunk_size <= 0 or max_chunk_size > DEFAULT_MAX_CHUNK_SIZE:
            raise IndexingError(
                "max_chunk_size must be between 1 and "
                f"{DEFAULT_MAX_CHUNK_SIZE} characters."
            )

        if not self.raw_dir.exists():
            raise IndexingError(
                f"Input directory not found: {self.raw_dir}"
            )

        if not self.raw_dir.is_dir():
            raise IndexingError(
                f"Expected a directory: {self.raw_dir}"
            )

        chunks: list[Chunk] = []
        files_indexed = 0
        files_skipped = 0

        for file_path in tqdm(
            self._iter_source_files(),
            desc="Indexing files",
            unit="file",
        ):
            try:
                content = file_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                print(f"Warning: skipping unreadable file {file_path}: {exc}")
                files_skipped += 1
                continue

            relative_path = self._to_project_relative_path(file_path)

            chunker = self._select_chunker(
                relative_path,
                max_chunk_size=max_chunk_size,
            )

            file_chunks = chunker.chunk_file(
                relative_path,
                content,
            )

            if file_chunks:
                chunks.extend(file_chunks)
                files_indexed += 1

        lexical_index = self.lexical_indexer.build(chunks)

        self.storage.save(
            chunks=chunks,
            lexical_index=lexical_index,
        )

        return {
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "chunks": len(chunks),
        }

    def _iter_source_files(self) -> list[Path]:
        """Return all source files under the raw directory."""
        if not self.raw_dir.exists():
            return []

        files: list[Path] = []

        for path in sorted(self.raw_dir.rglob("*")):
            if not path.is_file():
                continue

            if any(part in IGNORED_DIRS for part in path.parts):
                continue

            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue

            files.append(path)

        return files

    @staticmethod
    def _select_chunker(
        relative_path: str,
        max_chunk_size: int,
    ):
        """Choose the chunker according to the file type."""
        suffix = Path(relative_path).suffix.lower()

        if suffix == ".py":
            return PythonChunker(
                max_chunk_size=max_chunk_size,
            )

        return MarkdownChunker(
            max_chunk_size=max_chunk_size,
        )

    @staticmethod
    def _to_project_relative_path(
        file_path: Path,
    ) -> str:
        """
        Return the path relative to the project root.
        """
        try:
            return file_path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise IndexingError(
                f"File is outside the project root: {file_path}"
            ) from exc
