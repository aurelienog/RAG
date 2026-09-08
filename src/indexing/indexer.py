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
        """Index source files into searchable chunks.

        The indexer discovers supported source files, selects an appropriate
        chunker for each file type, builds a lexical index from the resulting
        chunks, and persists the generated index.

        Attributes:
            raw_dir: Directory containing the source files to index.
            storage: Storage backend used to persist chunks and the lexical index.
            lexical_indexer: Component used to build the lexical index.
        """
        self.raw_dir = Path(raw_dir)
        self.storage = IndexStorage(processed_dir)
        self.lexical_indexer = LexicalIndexer()

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, int]:
        """Index supported source files and persist the resulting index.

        Files with unsupported extensions or ignored directory components are
        skipped during discovery. Python files are processed with a
        ``PythonChunker``, while other supported files are processed with a
        ``MarkdownChunker``.

        Unreadable files are skipped and counted in the returned statistics.

        Args:
            max_chunk_size: Maximum number of characters allowed in each
                generated chunk.

        Returns:
            A dictionary containing the number of indexed files, the number
            of skipped files, and the total number of generated chunks. The
            keys are ``files_indexed``, ``files_skipped``, and ``chunks``.

        Raises:
            IndexingError: If ``max_chunk_size`` is outside the allowed
                range, if the input directory does not exist, or if the input
                path is not a directory.
        """

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
        """Find all supported source files under the raw directory.

        Files located inside ignored directories or files whose extensions
        are not included in ``ALLOWED_SUFFIXES`` are excluded.

        Returns:
            A sorted list of supported source file paths. An empty list is
            returned if the raw directory does not exist.
        """

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
        """Select a chunker based on the source file extension.

        Python files use ``PythonChunker``. Other supported file types use
        ``MarkdownChunker``.

        Args:
            relative_path: Source file path relative to the project root.
            max_chunk_size: Maximum number of characters allowed in each
                generated chunk.

        Returns:
            A configured chunker appropriate for the file type.
        """
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
        """Convert a file path to a project-relative POSIX path.

        Args:
            file_path: Path to the source file.

        Returns:
            The file path relative to the project root, using POSIX
            separators.

        Raises:
            IndexingError: If the file is outside the configured project
                root.
        """
        try:
            return file_path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise IndexingError(
                f"File is outside the project root: {file_path}"
            ) from exc
