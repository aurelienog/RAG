from abc import ABC, abstractmethod

from ...domain import Chunk, IndexingError
from ...config import DEFAULT_MAX_CHUNK_SIZE


class BaseChunker(ABC):
    """
    Common interface for source chunkers.
    """

    def __init__(self, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> None:
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            raise IndexingError("max_chunk_size must be between 1 and 2000 characters.")

        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        """
        Split source content into indexable chunks.

        Implementations must guarantee that every returned chunk
        respects max_chunk_size.
        """
        raise NotImplementedError
