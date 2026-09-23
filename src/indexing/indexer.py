import hashlib
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
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


class FileChangeSummary(TypedDict):
    """Typed structure describing which files changed since the previous index."""

    unchanged: set[str]
    modified: set[str]
    new: set[str]
    deleted: set[str]
    hashes: dict[str, str]


class IndexStats(TypedDict):
    """Typed summary returned after a full indexing pass."""

    files_indexed: int
    files_skipped: int
    chunks: int
    unchanged_files: int
    modified_files: int
    new_files: int
    deleted_files: int


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
        build_semantic_embeddings: bool = False,
    ) -> IndexStats:
        """Index supported source files and persist the resulting index.

        Files with unsupported extensions or ignored directory components are
        skipped during discovery. Python files are processed with a
        ``PythonChunker``, while other supported files are processed with a
        ``MarkdownChunker``.

        Unreadable files are skipped and counted in the returned statistics.

        Args:
            max_chunk_size: Maximum number of characters allowed in each
                generated chunk.
            build_semantic_embeddings: When ``True``, additionally create a
                lightweight CPU embedding matrix for semantic retrieval. This
                is intentionally disabled by default because it loads the
                SentenceTransformers model and is much slower than lexical
                indexing alone.

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

        index_file = self.storage.processed_dir / "index.json"
        if not index_file.exists():
            old_manifest = {}
            old_chunks_by_file = {}
        else:
            try:
                old_manifest = self._load_manifest()
                old_chunks_by_file = self._load_previous_chunks()
            except IndexingError:
                old_manifest = {}
                old_chunks_by_file = {}

        current_files = self._iter_source_files()
        changes: FileChangeSummary = self._detect_changes(
            current_files,
            old_manifest,
        )

        chunks: list[Chunk] = []
        files_indexed = 0
        files_skipped = 0
        unchanged_files = 0
        modified_files = 0
        new_files = 0
        deleted_files = len(changes["deleted"])
        next_manifest: dict[str, dict[str, object]] = {}

        for file_path in tqdm(
            current_files,
            desc="Indexing files",
            unit="file",
        ):
            relative_path = self._to_project_relative_path(file_path)

            if relative_path in changes["unchanged"]:
                reused_chunks = self._reuse_unchanged_chunks(
                    relative_path,
                    old_chunks_by_file.get(relative_path, []),
                )
                chunks.extend(reused_chunks)
                files_indexed += 1
                unchanged_files += 1
                current_hash = changes["hashes"][relative_path]
                next_manifest[relative_path] = {
                    "hash": current_hash,
                    "chunk_ids": [chunk.id for chunk in reused_chunks],
                }
                continue

            try:
                content = file_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                print(f"Warning: skipping unreadable file {file_path}: {exc}")
                files_skipped += 1
                continue

            file_chunks = self._chunk_file(
                file_path,
                relative_path,
                content,
                max_chunk_size=max_chunk_size,
            )

            if file_chunks:
                chunks.extend(file_chunks)
                files_indexed += 1

            if relative_path in changes["modified"]:
                modified_files += 1
            else:
                new_files += 1

            current_hash = changes["hashes"][relative_path]
            next_manifest[relative_path] = {
                "hash": current_hash,
                "chunk_ids": [chunk.id for chunk in file_chunks],
            }

        lexical_index = self.lexical_indexer.build(chunks)

        embeddings = self._update_embeddings(
            chunks,
            old_manifest=old_manifest,
            new_manifest=next_manifest,
            build_semantic_embeddings=build_semantic_embeddings,
        )

        self.storage.save(
            chunks=chunks,
            lexical_index=lexical_index,
            embeddings=embeddings,
        )
        self._save_manifest(next_manifest)

        return {
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "chunks": len(chunks),
            "unchanged_files": unchanged_files,
            "modified_files": modified_files,
            "new_files": new_files,
            "deleted_files": deleted_files,
        }

    def _get_file_hash(self, file_path: Path) -> str:
        """Return the SHA-256 hash for the file payload.

        Args:
            file_path: The path to the target file.

        Returns:
            The hexadecimal SHA-256 hash string of the file content.
        """
        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                digest.update(chunk)

        return digest.hexdigest()

    def _load_manifest(self) -> dict[str, dict[str, object]]:
        """Load the previously persisted manifest of indexed files.

        Returns:
            A dictionary mapping file paths to their historical metadata
            (such as file hashes and associated chunk IDs).
        """
        return self.storage.load_manifest()

    def _save_manifest(self, manifest: dict[str, dict[str, object]]) -> None:
        """Persist the file manifest describing the current index state.

        Args:
            manifest: The dictionary containing the current file paths,
                hashes, and chunk IDs to be written to storage.
        """
        self.storage.save_manifest(manifest)

    def _detect_changes(
        self,
        current_files: list[Path],
        old_manifest: dict[str, dict[str, object]],
    ) -> FileChangeSummary:
        """Compare current files against the old manifest to detect file modifications.

        Args:
            current_files: A list of Paths pointing to the discovered files
                on the file system.
            old_manifest: The manifest structure representing the index state
                from the previous run.

        Returns:
            A FileChangeSummary describing categorized files into unchanged,
            modified, new, or deleted statuses, alongside their fresh hashes.
        """

        current_paths = {
            self._to_project_relative_path(file_path)
            for file_path in current_files
        }

        deleted_paths = set(old_manifest) - current_paths

        unchanged: set[str] = set()
        modified: set[str] = set()
        new: set[str] = set()
        current_hashes: dict[str, str] = {}

        for file_path in current_files:
            relative_path = self._to_project_relative_path(file_path)
            current_hash = self._get_file_hash(file_path)

            current_hashes[relative_path] = current_hash

            previous_entry = old_manifest.get(relative_path)

            if (
                previous_entry is not None
                and previous_entry.get("hash") == current_hash
            ):
                unchanged.add(relative_path)
            elif previous_entry is not None:
                modified.add(relative_path)
            else:
                new.add(relative_path)

        return {
            "unchanged": unchanged,
            "modified": modified,
            "new": new,
            "deleted": deleted_paths,
            "hashes": current_hashes,
        }

    def _chunk_file(
        self,
        file_path: Path,
        relative_path: str,
        content: str,
        max_chunk_size: int,
    ) -> list[Chunk]:
        """Create chunks for a specific file, selecting the correct chunker.

        Args:
            file_path: The absolute or raw Path to the target file.
            relative_path: The project-relative POSIX path of the file.
            content: The raw text content of the file to be chunked.
            max_chunk_size: Maximum number of characters allowed per chunk.

        Returns:
            A list of generated Chunk objects for the given file.
        """
        chunker = self._select_chunker(
            relative_path,
            max_chunk_size=max_chunk_size,
        )

        return chunker.chunk_file(
            relative_path,
            content,
        )

    def _reuse_unchanged_chunks(
        self,
        file_path: str,
        old_chunks: list[Chunk],
    ) -> list[Chunk]:
        """Reuse a file's previous chunks when the content hash is unchanged.

        Args:
            file_path: The project-relative POSIX path of the file.
            old_chunks: A list of Chunk objects generated in a previous run.

        Returns:
            A shallow copy list containing the reused Chunk objects.
        """
        return list(old_chunks)

    def _update_embeddings(
        self,
        chunks: list[Chunk],
        old_manifest: dict[str, dict[str, object]] | None = None,
        new_manifest: dict[str, dict[str, object]] | None = None,
        build_semantic_embeddings: bool = False,
    ) -> np.ndarray | None:
        """Persist semantic embeddings incrementally by reusing stable chunk IDs.

        This method identifies which files were added, modified, or deleted,
        and updates the embedding matrix accordingly. Stable chunks skip
        recomputation to improve execution speed.

        Args:
            chunks: A list of all active Chunk objects for the current index pass.
            old_manifest: The file manifest metadata from the previous run.
                Defaults to None.
            new_manifest: The fresh file manifest metadata generated in the
                current run. Defaults to None.
            build_semantic_embeddings: When True, triggers semantic embedding
                generation via SentenceTransformers. Defaults to False.

        Returns:
            A NumPy ndarray containing the stacked vector embeddings for all
            active chunks, or None if build_semantic_embeddings is disabled or
            the retriever fails to load.
        """
        if not build_semantic_embeddings:
            return None

        try:
            from ..retrieval.semantic_retriever import SemanticRetriever
        except Exception as exc:  # pragma: no cover - optional CPU fallback
            print(f"Warning: semantic embeddings were skipped: {exc}")
            return None

        existing_embeddings, existing_ids = (
            self.storage.load_embeddings_with_ids()
        )

        if existing_embeddings is None or existing_ids is None:
            embeddings = SemanticRetriever.build_embeddings(chunks)
            self.storage.save_embeddings(
                embeddings,
                chunk_ids=[chunk.id for chunk in chunks],
            )
            return embeddings

        old_manifest = old_manifest or {}
        new_manifest = new_manifest or {}

        ids_to_recompute: set[str] = set()
        ids_to_remove: set[str] = set()

        for file_path, entry in old_manifest.items():
            if file_path not in new_manifest:
                ids_to_remove.update(cast(list[str], entry.get("chunk_ids", [])))

        for file_path, entry in new_manifest.items():
            previous_entry = old_manifest.get(file_path)
            current_chunk_ids = set(cast(list[str], entry.get("chunk_ids", [])))

            if previous_entry is None:
                ids_to_recompute.update(current_chunk_ids)
                continue

            previous_chunk_ids = set(cast(list[str], previous_entry.get("chunk_ids", [])))
            previous_hash = previous_entry.get("hash")
            current_hash = entry.get("hash")

            if previous_hash != current_hash:
                ids_to_remove.update(previous_chunk_ids)
                ids_to_recompute.update(current_chunk_ids)

        current_chunk_ids = {chunk.id for chunk in chunks}
        ids_to_recompute.update(current_chunk_ids - set(existing_ids))

        embedding_map: dict[str, np.ndarray] = {}
        for chunk_id, vector in zip(existing_ids, existing_embeddings):
            if chunk_id in ids_to_remove or chunk_id in ids_to_recompute:
                continue
            embedding_map[chunk_id] = np.asarray(vector, dtype=np.float32)

        if ids_to_recompute:
            recompute_chunks = [
                chunk for chunk in chunks if chunk.id in ids_to_recompute
            ]
            if recompute_chunks:
                recomputed = SemanticRetriever.build_embeddings(recompute_chunks)
                for chunk, vector in zip(recompute_chunks, recomputed):
                    embedding_map[chunk.id] = np.asarray(vector, dtype=np.float32)

        final_ids = [chunk.id for chunk in chunks]
        missing_ids = [chunk_id for chunk_id in final_ids if chunk_id not in embedding_map]

        if missing_ids:
            missing_chunks = [
                chunk for chunk in chunks if chunk.id in missing_ids
            ]
            missing_vectors = SemanticRetriever.build_embeddings(missing_chunks)
            for chunk, vector in zip(missing_chunks, missing_vectors):
                embedding_map[chunk.id] = np.asarray(vector, dtype=np.float32)

        if not final_ids:
            final_embeddings = np.empty((0, 0), dtype=np.float32)
            self.storage.save_embeddings(
                final_embeddings,
                chunk_ids=[],
            )
            return final_embeddings

        final_embeddings = np.vstack([
            embedding_map[chunk_id]
            for chunk_id in final_ids
        ])

        self.storage.save_embeddings(
            final_embeddings,
            chunk_ids=final_ids,
        )

        return final_embeddings

    def _load_previous_chunks(self) -> dict[str, list[Chunk]]:
        """Load previously persisted chunks keyed by file path.

        Returns:
            A dictionary mapping file path strings to their corresponding
            list of historical Chunk objects. Returns an empty dict if the
            storage loading fails or throws an IndexingError.
        """
        try:
            previous_chunks, _ = self.storage.load()
        except IndexingError:
            return {}

        grouped: dict[str, list[Chunk]] = {}
        for chunk in previous_chunks:
            grouped.setdefault(chunk.file_path, []).append(chunk)

        return grouped

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
    ) -> PythonChunker | MarkdownChunker:
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
