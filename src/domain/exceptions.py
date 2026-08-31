class RAGError(Exception):
    """Base exception for the RAG system."""
    pass


class IndexingError(RAGError):
    """Raised when indexing fails."""
    pass


class RetrievalError(RAGError):
    """Raised when retrieval fails."""
    pass


class DatasetError(RAGError):
    """Raised when a dataset cannot be loaded or validated."""
    pass


class GenerationError(RAGError):
    """Raised when answer generation fails."""
    pass


class EvaluationError(RAGError):
    """Raised when retrieval evaluation fails."""
    pass
