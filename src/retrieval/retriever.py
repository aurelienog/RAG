from typing import Protocol
from ..domain import Chunk


class Retriever(Protocol):
    """Protocol implemented by any retriever used by the search service."""
    chunks: list[Chunk]

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """Retrieve the top-k chunks for a query.

        Args:
            query: The text content or question to search for.
            k: The number of high-quality chunks to return. Defaults to 10.

        Returns:
            A list containing up to k matching Chunk objects sorted by
            relevance in descending order.
        """
        ...
