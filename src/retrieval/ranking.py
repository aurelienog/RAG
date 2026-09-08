import math

from ..domain import RetrievalError


def calculate_idf(
    total_docs: int,
    doc_freq: int,
) -> float:
    """Calculate the inverse document frequency used by BM25.

    The IDF value measures how informative a term is based on the number of
    documents in the collection and the number of documents containing the
    term.

    Args:
        total_docs: Total number of documents in the collection.
        doc_freq: Number of documents containing the term.

    Returns:
        The BM25 inverse document frequency value for the term.

    Raises:
        RetrievalError: If ``total_docs`` or ``doc_freq`` is not greater than
            zero, or if ``doc_freq`` is greater than ``total_docs``.
    """

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
    """Calculate the BM25 score contribution of one term in one document.

    The score combines term frequency, document length normalization, and
    inverse document frequency according to the BM25 ranking formula.

    Args:
        term_freq: Number of occurrences of the term in the document.
        doc_length: Number of tokens in the document.
        avg_doc_length: Average number of tokens across all documents.
        idf: Inverse document frequency of the term.
        k1: Controls the saturation of the term frequency contribution.
        b: Controls the strength of document length normalization.

    Returns:
        The BM25 score contribution of the term for the document.

    Raises:
        RetrievalError: If ``term_freq`` is not greater than zero,
            ``doc_length`` is negative, ``avg_doc_length`` is not greater
            than zero, ``k1`` is negative, or ``b`` is outside the range
            from 0 to 1.
    """

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
