from typing import Protocol
from ..domain import Chunk


class Retriever(Protocol):
    """Protocol implemented by any retriever used by the search service."""

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """Retrieve the top-k chunks for a query."""
        ...
