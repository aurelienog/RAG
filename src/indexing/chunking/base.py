from abc import ABC, abstractmethod

from ...domain import Chunk, IndexingError
from ...config import DEFAULT_MAX_CHUNK_SIZE


class BaseChunker(ABC):
    """Define the common interface and constraints for source chunkers.

    Implementations are responsible for splitting source files into
    indexable chunks while ensuring that each chunk respects the configured
    maximum chunk size.

    Attributes:
        max_chunk_size: Maximum number of characters allowed in each chunk.
    """

    def __init__(self, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> None:
        """Initialize a chunker with a maximum chunk size.

        Args:
            max_chunk_size: Maximum number of characters allowed in each
                generated chunk.

        Raises:
            IndexingError: If ``max_chunk_size`` is less than 1 or greater
                than the configured maximum chunk size.
        """
        if max_chunk_size <= 0 or max_chunk_size > DEFAULT_MAX_CHUNK_SIZE:
            raise IndexingError("max_chunk_size must be between 1 and "
                                f"{DEFAULT_MAX_CHUNK_SIZE} characters.")

        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        """Split source content into indexable chunks.

        Implementations must guarantee that every returned chunk respects
        ``max_chunk_size``.

        Args:
            file_path: Path of the source file being chunked.
            content: Source file content to split into chunks.

        Returns:
            A list of chunks created from the source content.
        """
        raise NotImplementedError
