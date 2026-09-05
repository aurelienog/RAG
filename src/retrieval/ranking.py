import math

from ..domain import RetrievalError


def calculate_idf(
    total_docs: int,
    doc_freq: int,
) -> float:
    """Calculate the BM25 inverse document frequency."""

    if total_docs <= 0:
        raise RetrievalError(
            "total_docs must be greater than zero."
        )

    if doc_freq <= 0:
        raise RetrievalError(
            "doc_freq must be greater than zero."
        )

    if doc_freq > total_docs:
        raise RetrievalError(
            "doc_freq cannot be greater than total_docs."
        )

    numerator = total_docs - doc_freq + 0.5
    denominator = doc_freq + 0.5

    return math.log(
        1.0 + (numerator / denominator)
    )


def score_bm25_term(
    term_freq: int,
    doc_length: int,
    avg_doc_length: float,
    idf: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Calculate the BM25 score for one term in one document."""

    if term_freq <= 0:
        raise RetrievalError(
            "term_freq must be greater than zero."
        )

    if doc_length < 0:
        raise RetrievalError(
            "doc_length cannot be negative."
        )

    if avg_doc_length <= 0:
        raise RetrievalError(
            "avg_doc_length must be greater than zero."
        )

    if k1 < 0:
        raise RetrievalError(
            "k1 cannot be negative."
        )

    if not 0.0 <= b <= 1.0:
        raise RetrievalError(
            "b must be between 0 and 1."
        )

    length_normalization = (
        1.0
        - b
        + b * (doc_length / avg_doc_length)
    )

    numerator = term_freq * (k1 + 1.0)

    denominator = (
        term_freq
        + k1 * length_normalization
    )

    return idf * (numerator / denominator)
