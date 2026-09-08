from collections import Counter
from dataclasses import dataclass
from typing import TypedDict

from tqdm import tqdm

from ..domain import Chunk
from ..utils import Tokenizer


class Posting(TypedDict):
    """Represent a term occurrence within a document chunk.

    Attributes:
        chunk_id: Unique identifier of the chunk containing the term.
        tf: Number of times the term occurs in the chunk.
    """
    chunk_id: str
    tf: int


@dataclass
class LexicalIndex:
    """Store lexical data required for BM25 retrieval.

    The index contains an inverted index mapping terms to the chunks in
    which they occur, together with document frequency and document length
    statistics used by the BM25 scoring algorithm.

    Attributes:
        inverted_index: Mapping from each term to its postings list.
        doc_freq: Number of chunks containing each term.
        doc_lengths: Number of tokens in each chunk.
        avg_doc_length: Average number of tokens per chunk.
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
        """Build a lexical index from the provided chunks.

        Each chunk is tokenized and its term frequencies are calculated.
        The resulting postings, document frequencies, document lengths, and
        average document length are stored in a ``LexicalIndex``.

        Args:
            chunks: Source chunks to tokenize and include in the lexical
                index.

        Returns:
            A lexical index containing the inverted index and document
            statistics required for BM25 retrieval.
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
