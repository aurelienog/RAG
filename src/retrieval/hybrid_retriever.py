from __future__ import annotations

from ..domain import Chunk
from .bm25_retriever import BM25Retriever
from .semantic_retriever import SemanticRetriever


class HybridRetriever:
    """Fuse BM25 and semantic retrieval using Reciprocal Rank Fusion (RRF).

    Attributes:
        bm25_retriever: Component handling keyword-based BM25 retrieval.
        semantic_retriever: Component handling vector-based dense retrieval.
        rrf_k: A constant factor that penalizes low-ranked items in the
        RRF scoring formula.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            bm25_retriever: The lexical search engine instance.
            semantic_retriever: The vector search engine instance.
            rrf_k: Ranking constant used to tune the influence of top ranks
                during score calculation. Defaults to 60.
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []

    def search(
        self,
        query: str,
        k: int = 10,
        lexical_k: int | None = None,
        semantic_k: int | None = None,
    ) -> list[Chunk]:
        """Combine lexical and semantic retrieval into a single ranked list.

        This method requests separate candidate lists from both retrievers and
        re-ranks the unified results using Reciprocal Rank Fusion (RRF).

        Args:
            query: The text content or question to search for.
            k: The final number of high-quality chunks to return. Defaults to 10.
            lexical_k: The number of candidate chunks to pull from the BM25
                retriever. If None, it scales automatically based on k.
            semantic_k: The number of candidate chunks to pull from the semantic
                retriever. If None, it scales automatically based on k.

        Returns:
            A list containing up to k top-ranked Chunk objects sorted by
            their merged RRF score in descending order.

        Raises:
            ValueError: If the query string is empty or contains only whitespace,
                or if k is equal to or less than 0.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        lexical_k = lexical_k or max(k * 4, 10)
        semantic_k = semantic_k or max(k * 4, 10)

        bm25_hits = self.bm25_retriever.search(query, k=lexical_k)
        semantic_hits = self.semantic_retriever.search(query, k=semantic_k)

        if not bm25_hits and not semantic_hits:
            return []

        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, chunk in enumerate(bm25_hits):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(semantic_hits):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )
            chunk_map[chunk.id] = chunk

        ordered_ids = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]

        return [chunk_map[chunk_id] for chunk_id, _ in ordered_ids]
