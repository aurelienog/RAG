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
        self.embeddings, self.embedding_ids = self.storage.load_embeddings_with_ids()
        self.chunk_map = {chunk.id: chunk for chunk in self.chunks}

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
        """Encode all chunk texts into a dense embedding matrix with tqdm progress."""
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RetrievalError(
                "sentence-transformers is required for semantic retrieval. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        if model is None:
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
                all_vectors.append(np.asarray(batch_vectors, dtype=np.float32))
                progress.update(len(batch))

        return np.vstack(all_vectors)

    def _ensure_embeddings(self) -> tuple[np.ndarray, list[str]]:
        if self.embeddings is not None:
            if self.embedding_ids is None:
                self.embedding_ids = [chunk.id for chunk in self.chunks]
            return self.embeddings, self.embedding_ids

        embeddings, chunk_ids = self.storage.load_embeddings_with_ids()
        if embeddings is not None:
            self.embeddings = embeddings
            self.embedding_ids = chunk_ids or [chunk.id for chunk in self.chunks]
            return self.embeddings, self.embedding_ids

        self.embeddings = self.build_embeddings(
            self.chunks,
            model_name=self.model_name,
            model=self.model,
        )
        self.embedding_ids = [chunk.id for chunk in self.chunks]
        self.storage.save_embeddings(
            self.embeddings,
            chunk_ids=self.embedding_ids,
        )
        return self.embeddings, self.embedding_ids

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """Return the top-k chunks ranked by semantic similarity."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not self.chunks:
            return []

        embeddings, embedding_ids = self._ensure_embeddings()
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

        similarities = embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:k]

        results: list[Chunk] = []
        for idx in top_indices:
            chunk_id = embedding_ids[int(idx)]
            chunk = self.chunk_map.get(chunk_id)
            if chunk is not None:
                results.append(chunk)

        return results
