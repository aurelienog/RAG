from pathlib import Path
from typing import List

from ..config import DATA_PROCESSED
from ..domain import Chunk
from ..utils import Tokenizer
from .storage import IndexStorage
from .ranking import calculate_idf, score_bm25_term


class Retriever:
    """
    Orchestrates loading the static lexical index and calculating 
    the top-k relevant snippets using BM25 rankings.
    """

    def __init__(self, processed_dir: str | Path = DATA_PROCESSED) -> None:
        self.storage = IndexStorage(processed_dir)
        self.chunks, self.lexical_index = self.storage.load()

        # Mapeamos los chunks por su ID para un acceso instantáneo O(1) al final
        self.chunk_map = {chunk.id: chunk for chunk in self.chunks}
        self.total_docs = len(self.chunks)

    def search(self, query: str, k: int = 10) -> List[Chunk]:
        """
        Retorna los top-k Chunks más relevantes ordenados descendentemente.
        """
        # Sanitizar entradas de borde exigido por el subject
        if not query or not query.strip():
            return []

        # 1. Tokenizar la consulta del usuario usando el mismo Tokenizer del indexador
        query_tokens = Tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Diccionario para acumular los scores finales de cada chunk candidato: {chunk_id: score_total}
        chunk_scores: dict[str, float] = {}

        # 2. Iterar término por término de la query
        for term in query_tokens:
            # Si el término no existe en nuestro vocabulario, pasamos al siguiente
            if term not in self.lexical_index.doc_freq:
                continue

            # Obtener precalculados estadísticos del término
            df = self.lexical_index.doc_freq[term]
            idf = calculate_idf(self.total_docs, df)

            # Obtener la lista de todos los chunks que contienen este término
            postings = self.lexical_index.inverted_index[term]

            # 3. Puntuar cada uno de los chunks que contienen el término actual
            for posting in postings:
                chunk_id = posting["chunk_id"]
                tf = posting["tf"]

                # Obtener la longitud específica de este chunk
                doc_len = self.lexical_index.doc_lengths[chunk_id]

                # Calcular el score parcial de este término en este chunk
                term_score = score_bm25_term(
                    term_freq=tf,
                    doc_length=doc_len,
                    avg_doc_length=self.lexical_index.avg_doc_length,
                    idf=idf
                )

                # Acumular el score en el chunk correspondiente
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + term_score

        # 4. Si ningún término coincidió con el índice, retornamos una lista vacía
        if not chunk_scores:
            return []

        # 5. Ordenar los IDs de los chunks candidatos por su puntuación acumulada descendentemente
        sorted_candidates = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)

        # 6. Tomar los primeros k resultados y resolver los objetos Chunk completos
        top_k_chunks = [
            self.chunk_map[chunk_id]
            for chunk_id, _ in sorted_candidates[:min(k, len(sorted_candidates))]
        ]

        return top_k_chunks

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
