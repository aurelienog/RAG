from pathlib import Path

from ..config import DATA_PROCESSED
from ..domain import Chunk, RetrievalError
from ..indexing.storage import IndexStorage
from ..utils import Tokenizer
from .ranking import calculate_idf, score_bm25_term


class Retriever:
    """Retrieve the most relevant chunks from a lexical index using BM25.

    The retriever loads a persisted lexical index and its corresponding
    chunks, then ranks candidate chunks according to their BM25 scores.

    Attributes:
        storage: Storage backend used to load the persisted index.
        chunks: All chunks available in the loaded index.
        lexical_index: Lexical data used for BM25 scoring.
        chunk_map: Mapping from chunk identifiers to their corresponding
            ``Chunk`` objects.
        total_docs: Total number of chunks in the loaded index.
    """

    def __init__(
        self,
        processed_dir: str | Path = DATA_PROCESSED,
    ) -> None:
        """Initialize the retriever and load the persisted index.

        Args:
            processed_dir: Directory containing the persisted lexical index.

        Raises:
            IndexingError: If the index cannot be loaded or has an invalid
                format.
        """
        self.storage = IndexStorage(processed_dir)
        self.chunks, self.lexical_index = self.storage.load()

        self.chunk_map = {
            chunk.id: chunk
            for chunk in self.chunks
        }

        self.total_docs = len(self.chunks)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """Retrieve the top-k chunks ranked by BM25 relevance.

        The query is tokenized and each known query term contributes to the
        BM25 score of the chunks containing that term. Results are returned
        in descending order of their accumulated BM25 score.

        Args:
            query: Search query used to retrieve relevant chunks.
            k: Maximum number of chunks to return.

        Returns:
            A list containing up to ``k`` chunks ordered from most to least
            relevant. An empty list is returned when the query contains no
            recognized tokens, the index contains no documents, or no query
            terms occur in the index.

        Raises:
            ValueError: If ``query`` is empty or ``k`` is not greater than
                zero.
            RetrievalError: If the lexical index references a chunk without
                a document length or references a chunk that cannot be found
                in the loaded chunk map.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        query_tokens = Tokenizer.tokenize(query)

        if not query_tokens:
            return []

        if self.total_docs == 0:
            return []

        chunk_scores: dict[str, float] = {}

        for term in query_tokens:
            df = self.lexical_index.doc_freq.get(term)

            if df is None:
                continue

            idf = calculate_idf(
                self.total_docs,
                df,
            )

            postings = self.lexical_index.inverted_index.get(
                term,
                [],
            )

            for posting in postings:
                chunk_id = posting["chunk_id"]
                tf = posting["tf"]

                doc_length = self.lexical_index.doc_lengths.get(
                    chunk_id
                )

                if doc_length is None:
                    raise RetrievalError(
                        f"Invalid index: missing document length "
                        f"for chunk '{chunk_id}'."
                    )

                term_score = score_bm25_term(
                    term_freq=tf,
                    doc_length=doc_length,
                    avg_doc_length=self.lexical_index.avg_doc_length,
                    idf=idf,
                )

                chunk_scores[chunk_id] = (
                    chunk_scores.get(chunk_id, 0.0)
                    + term_score
                )

        if not chunk_scores:
            return []

        sorted_candidates = sorted(
            chunk_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[Chunk] = []

        for chunk_id, _ in sorted_candidates[:k]:
            chunk = self.chunk_map.get(chunk_id)

            if chunk is None:
                raise RetrievalError(
                    f"Invalid index: chunk '{chunk_id}' not found."
                )

            results.append(chunk)

        return results

# def search_hybrid(self, query: str, k: int = 10, rrf_k: int = 60) -> list[Chunk]:
#     """
#     Combina los resultados léxicos (BM25) y semánticos (Embeddings)
#     usando Reciprocal Rank Fusion (RRF).
#     """
#     # 1. Obtener listas ordenadas crudas de ambos mundos
#     bm25_hits = self.search_bm25(query, k=k*4) # Pedimos de más para fusionar con margen
#     dense_hits = self.search_dense(query, k=k*4)

#     rrf_scores: dict[str, float] = {}

#     # 2. Aplicar penalización por posición a los resultados BM25
#     for rank, chunk in enumerate(bm25_hits):
#         # Fórmula estándar RRF: 1 / (k + rank + 1)
#         rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank + 1)

#     # 3. Aplicar penalización por posición a los resultados Semánticos
#     for rank, chunk in enumerate(dense_hits):
#         rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank + 1)

#     # 4. Ordenar todos los chunks unificados de mayor a menor score RRF
#     # Mapeamos los IDs ganadores de regreso a sus objetos Chunk reales
#     chunk_map = {c.id: c for c in (bm25_hits + dense_hits)}
#     sorted_ids = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]

#     return [chunk_map[chunk_id] for chunk_id, _ in sorted_ids]
