from collections import Counter
from dataclasses import dataclass
from typing import TypedDict

from tqdm import tqdm

from ..domain import Chunk
from ..utils import Tokenizer


class Posting(TypedDict):
    chunk_id: str
    tf: int


@dataclass
class LexicalIndex:
    """
    Lexical data required for BM25 retrieval.
    """

    inverted_index: dict[str, list[Posting]]
    doc_freq: dict[str, int]
    doc_lengths: dict[str, int]
    avg_doc_length: float


class LexicalIndexer:
    """
    Build the lexical index used by the BM25 retriever.
    """

    def build(self, chunks: list[Chunk]) -> LexicalIndex:
        """
        Build a lexical index from the provided chunks.
        """
        inverted_index: dict[str, list[Posting]] = {}
        doc_freq: dict[str, int] = {}
        doc_lengths: dict[str, int] = {}

        for chunk in tqdm(chunks, desc="Tokenizing chunks", unit="chunk"):
            tokens = Tokenizer.tokenize(chunk.text)
            doc_lengths[chunk.id] = len(tokens)

            term_frequencies = Counter(tokens)

            for term, frequency in term_frequencies.items():
                inverted_index.setdefault(term, []).append(
                    {
                        "chunk_id": chunk.id,
                        "tf": frequency,
                    }
                )

                doc_freq[term] = doc_freq.get(term, 0) + 1

        if doc_lengths:
            avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)
        else:
            avg_doc_length = 0.0

        return LexicalIndex(
            inverted_index=inverted_index,
            doc_freq=doc_freq,
            doc_lengths=doc_lengths,
            avg_doc_length=avg_doc_length,
        )
