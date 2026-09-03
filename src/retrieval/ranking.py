import math


def calculate_idf(total_docs: int, doc_freq: int) -> float:
    """
    Calcula la Frecuencia Inversa de Documento (IDF) usando la variante BM25.
    Evita valores negativos sumando 1 al argumento del logaritmo.
    """
    numerator = total_docs - doc_freq + 0.5
    denominator = doc_freq + 0.5
    # Nos aseguramos de no pasar un valor menor o igual a cero al logaritmo
    return math.log(max(numerator / denominator, 0.0) + 1.0)


def score_bm25_term(
    term_freq: int,
    doc_length: int,
    avg_doc_length: float,
    idf: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Calcula el score BM25 para un único término en un chunk específico.
    """
    if avg_doc_length == 0:
        return 0.0

    # Factor de normalización por longitud del documento
    length_normalization = 1.0 - b + (b * (doc_length / avg_doc_length))

    # Ecuación de saturación del Term Frequency (TF)
    numerator = term_freq * (k1 + 1.0)
    denominator = term_freq + (k1 * length_normalization)

    return idf * (numerator / denominator)
