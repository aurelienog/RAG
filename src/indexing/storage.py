import json
from pathlib import Path

import numpy as np

from ..domain import Chunk, IndexingError
from .lexical_index import LexicalIndex


class IndexStorage:
    """Persist and load the processed RAG index."""

    def __init__(self, processed_dir: str | Path) -> None:
        """Initialize the index storage.

        Args:
            processed_dir: Directory where the processed index is stored.
        """
        self.processed_dir = Path(processed_dir)

    def load_manifest(self) -> dict[str, dict[str, object]]:
        """Load the persisted file manifest used for incremental indexing.

        Returns:
            A dictionary mapping file paths to their cached properties (such as
            hashes and chunk IDs). Returns an empty dict if the file does not exist.

        Raises:
            IndexingError: If the manifest file cannot be read or contains
                malformed/invalid JSON data structures.
        """
        manifest_path = self.processed_dir / "manifest.json"

        if not manifest_path.exists():
            return {}

        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexingError(
                f"Could not read manifest: {manifest_path}"
            ) from exc

        if payload is None:
            return {}

        if not isinstance(payload, dict):
            raise IndexingError(
                f"Invalid manifest format: {manifest_path}"
            )

        return payload

    def save_manifest(self, manifest: dict[str, dict[str, object]]) -> None:
        """Persist the manifest describing the current file hash state.

        Args:
            manifest: A dictionary describing the current file path states,
                content hashes, and generated chunk references.

        Raises:
            IndexingError: If the manifest target path cannot be written to disk.
        """
        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = self.processed_dir / "manifest.json"

        try:
            with manifest_path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, sort_keys=True)
        except OSError as exc:
            raise IndexingError(
                f"Could not write manifest: {manifest_path}"
            ) from exc

    def save(
        self,
        chunks: list[Chunk],
        lexical_index: LexicalIndex,
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Save chunks and lexical index to JSON, with optional embeddings in NumPy.

        If embeddings are omitted or set to None, any stale embedding files
        previously found in the target directory are automatically unlinked.

        Args:
            chunks: A list of Chunk domain entities to save.
            lexical_index: The generated LexicalIndex structure holding frequency
                and statistics payloads.
            embeddings: An optional dense array matrix representing semantic vectors.
                Defaults to None.

        Raises:
            IndexingError: If writing the index structure or clean-up targets fails.
        """
        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "chunks": [
                {
                    "id": chunk.id,
                    "file_path": chunk.file_path,
                    "text": chunk.text,
                    "first_character_index": chunk.start,
                    "last_character_index": chunk.end,
                    "kind": chunk.kind,
                }
                for chunk in chunks
            ],
            "inverted_index": lexical_index.inverted_index,
            "doc_freq": lexical_index.doc_freq,
            "doc_lengths": lexical_index.doc_lengths,
            "avg_doc_length": lexical_index.avg_doc_length,
        }

        output_path = self.processed_dir / "index.json"

        try:
            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                )
        except OSError as exc:
            raise IndexingError(
                f"Could not write index: {output_path}"
            ) from exc

        if embeddings is not None:
            self.save_embeddings(
                embeddings,
                chunk_ids=[chunk.id for chunk in chunks],
            )
        else:
            # No conservar embeddings pertenecientes a un índice anterior.
            embeddings_path = self.processed_dir / "embeddings.npy"
            embeddings_ids_path = self.processed_dir / "embeddings_ids.json"

            embeddings_path.unlink(missing_ok=True)
            embeddings_ids_path.unlink(missing_ok=True)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str] | None = None,
    ) -> None:
        """Persist a dense embedding matrix and its chunk ids as separate files.

        Args:
            embeddings: A NumPy matrix holding the array representations of vectors.
            chunk_ids: An optional matching list of distinct ID strings mapped
                sequentially per row. Defaults to None.

        Raises:
            IndexingError: If the length of chunk_ids does not match row shapes or
                if writes to file system streams fail.
        """
        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self.processed_dir / "embeddings.npy"
        arr = np.asarray(embeddings, dtype=np.float32)

        if chunk_ids is not None and len(chunk_ids) != len(arr):
            raise IndexingError(
                "Embedding rows and chunk ids are out of sync: "
                f"{len(arr)} rows vs {len(chunk_ids)} ids."
            )

        try:
            np.save(output_path, arr)
        except (TypeError, ValueError, OSError) as exc:
            raise IndexingError(
                f"Could not write embeddings: {output_path}"
            ) from exc

        if chunk_ids is not None:
            metadata_path = self.processed_dir / "embeddings_ids.json"
            try:
                with metadata_path.open("w", encoding="utf-8") as file:
                    json.dump(chunk_ids, file)
            except OSError as exc:
                raise IndexingError(
                    f"Could not write embeddings metadata: {metadata_path}"
                ) from exc

    def load_embeddings(self) -> np.ndarray | None:
        """Load a persisted embedding matrix from ``embeddings.npy`` if present.

        Returns:
            The raw dense float32 array matrix if available, or None if the file
            is missing or empty.
        """
        embeddings, _ = self.load_embeddings_with_ids()
        return embeddings

    def load_embeddings_with_ids(self) -> tuple[np.ndarray | None, list[str] | None]:
        """Load embeddings and their explicit chunk-id order from storage.

        Returns:
            A tuple where the first element is the NumPy ndarray (or None) and
            the second element is the list of matching chunk ID strings (or None).

        Raises:
            IndexingError: If files are corrupted, metadata JSON structures are
                malformed, or row dimensions diverge between IDs and the array.
        """
        input_path = self.processed_dir / "embeddings.npy"

        if not input_path.exists():
            return None, None

        try:
            embeddings = np.load(input_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise IndexingError(
                f"Could not read embeddings: {input_path}"
            ) from exc

        if embeddings.size == 0:
            return None, None

        embeddings = np.asarray(embeddings, dtype=np.float32)
        metadata_path = self.processed_dir / "embeddings_ids.json"
        chunk_ids: list[str] | None = None

        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as file:
                    chunk_ids = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                raise IndexingError(
                    f"Could not read embeddings metadata: {metadata_path}"
                ) from exc

            if not isinstance(chunk_ids, list):
                raise IndexingError(
                    f"Invalid embeddings metadata: {metadata_path}"
                )

            if len(chunk_ids) != len(embeddings):
                raise IndexingError(
                    "Embedding rows and chunk ids are out of sync. "
                    "The metadata length does not match the stored matrix."
                )

        return embeddings, chunk_ids

    def load(self) -> tuple[list[Chunk], LexicalIndex]:
        """Load chunks and lexical index from the persisted JSON file.

        Returns:
            A tuple containing the list of indexed chunks and the lexical
            index reconstructed from the stored data.

        Raises:
            IndexingError: If the index file does not exist, cannot be read,
                contains invalid JSON, or does not match the expected index
                format.
        """
        input_path = self.processed_dir / "index.json"

        if not input_path.exists():
            raise IndexingError(
                f"Index file not found: {input_path}"
            )

        try:
            with input_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexingError(
                f"Could not read index: {input_path}"
            ) from exc

        try:
            chunks = [
                Chunk(
                    id=item["id"],
                    file_path=item["file_path"],
                    text=item["text"],
                    start=item["first_character_index"],
                    end=item["last_character_index"],
                    kind=item["kind"],
                )
                for item in payload["chunks"]
            ]

            lexical_index = LexicalIndex(
                inverted_index=payload["inverted_index"],
                doc_freq=payload["doc_freq"],
                doc_lengths=payload["doc_lengths"],
                avg_doc_length=payload["avg_doc_length"],
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise IndexingError(
                f"Invalid index format: {input_path}"
            ) from exc

        return chunks, lexical_index
