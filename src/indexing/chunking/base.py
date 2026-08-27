from abc import ABC, abstractmethod

from ...domain.chunk import Chunk


class BaseChunker(ABC):
    """
    Common interface for source chunkers.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0")

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