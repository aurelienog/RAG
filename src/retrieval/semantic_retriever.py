from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import DATA_PROCESSED
from ..domain import Chunk, RetrievalError
from ..indexing.storage import IndexStorage


class SemanticRetriever:
    """Retrieve chunks using cosine similarity over sentence embeddings."""

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        processed_dir: str | Path = DATA_PROCESSED,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.storage = IndexStorage(processed_dir)
        self.model_name = model_name

        self.model = self._load_model()

        self.chunks, _ = self.storage.load()

        self.chunk_map = {
            chunk.id: chunk
            for chunk in self.chunks
        }

        self.embeddings, self.embedding_ids = (
            self.storage.load_embeddings_with_ids()
        )

        self._validate_embedding_mapping()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RetrievalError(
                "sentence-transformers is required for semantic retrieval. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        return SentenceTransformer(
            self.model_name,
            device="cpu",
        )

    @staticmethod
    def build_embeddings(
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL_NAME,
        model=None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode all chunks into normalized dense embeddings."""

        if not chunks:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RetrievalError(
                    "sentence-transformers is required for semantic retrieval. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            model = SentenceTransformer(
                model_name,
                device="cpu",
            )

        texts = [chunk.text for chunk in chunks]
        all_vectors: list[np.ndarray] = []

        with tqdm(
            total=len(texts),
            desc="Encoding embeddings",
            unit="chunk",
        ) as progress:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]

                batch_vectors = model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )

                all_vectors.append(
                    np.asarray(
                        batch_vectors,
                        dtype=np.float32,
                    )
                )

                progress.update(len(batch))

        return np.vstack(all_vectors)

    def _validate_embedding_mapping(self) -> None:
        """
        Validate the explicit relationship:

            embedding[i] <-> embedding_ids[i] <-> chunk_map[id]

        We deliberately do not fall back to positional chunk ordering.
        """

        if self.embeddings is None:
            return

        if self.embedding_ids is None:
            raise RetrievalError(
                "Semantic embeddings exist but their chunk IDs are missing. "
                "Run 'index-semantic' again to rebuild the semantic index."
            )

        if self.embeddings.ndim != 2:
            raise RetrievalError(
                "Semantic embeddings must be a 2-dimensional matrix."
            )

        if len(self.embedding_ids) != len(self.embeddings):
            raise RetrievalError(
                "Semantic embeddings and chunk IDs are out of sync: "
                f"{len(self.embeddings)} rows vs "
                f"{len(self.embedding_ids)} IDs."
            )

        if len(self.embedding_ids) != len(set(self.embedding_ids)):
            raise RetrievalError(
                "Semantic embedding IDs are not unique."
            )

        missing_ids = [
            chunk_id
            for chunk_id in self.embedding_ids
            if chunk_id not in self.chunk_map
        ]

        if missing_ids:
            preview = ", ".join(missing_ids[:5])

            raise RetrievalError(
                "Semantic embeddings reference chunks that do not exist "
                f"in the current index: {preview}"
            )

    def _ensure_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """
        Return embeddings and their explicit chunk IDs.

        If the semantic index does not exist yet, build it once and persist
        both the matrix and its chunk-ID mapping.
        """

        if self.embeddings is not None:
            self._validate_embedding_mapping()

            # At this point validation guarantees that this is not None.
            assert self.embedding_ids is not None

            return self.embeddings, self.embedding_ids

        stored_embeddings, stored_ids = (
            self.storage.load_embeddings_with_ids()
        )

        if stored_embeddings is not None:
            self.embeddings = stored_embeddings
            self.embedding_ids = stored_ids

            self._validate_embedding_mapping()

            assert self.embedding_ids is not None

            return self.embeddings, self.embedding_ids

        if not self.chunks:
            self.embeddings = np.empty(
                (0, 0),
                dtype=np.float32,
            )
            self.embedding_ids = []

            return self.embeddings, self.embedding_ids

        self.embeddings = self.build_embeddings(
            self.chunks,
            model_name=self.model_name,
            model=self.model,
        )

        self.embedding_ids = [
            chunk.id
            for chunk in self.chunks
        ]

        self.storage.save_embeddings(
            self.embeddings,
            chunk_ids=self.embedding_ids,
        )

        self._validate_embedding_mapping()

        return self.embeddings, self.embedding_ids

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """Return the top-k chunks ranked by semantic similarity."""

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        if not self.chunks:
            return []

        embeddings, embedding_ids = (
            self._ensure_embeddings()
        )

        if embeddings.size == 0:
            return []

        query_embedding = np.asarray(
            self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        # Because both document and query embeddings are normalized,
        # the dot product is cosine similarity.
        similarities = embeddings @ query_embedding

        top_indices = np.argsort(
            similarities
        )[::-1][:k]

        results: list[Chunk] = []

        for idx in top_indices:
            chunk_id = embedding_ids[int(idx)]

            chunk = self.chunk_map.get(chunk_id)

            if chunk is None:
                raise RetrievalError(
                    "Semantic index references unknown chunk: "
                    f"{chunk_id}"
                )

            results.append(chunk)

        return results
