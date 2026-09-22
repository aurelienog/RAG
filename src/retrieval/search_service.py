import uuid
from pathlib import Path

from tqdm import tqdm

from ..domain import Chunk
from ..ingest import JsonStore
from ..models import MinimalSearchResults, StudentSearchResults, MinimalSource
from .retriever import Retriever


class SearchService:
    """Search indexed chunks and persist search results.

    This service coordinates retrieval through a ``Retriever`` and optional
    persistence through a ``JsonStore``.
    """

    def __init__(
        self,
        retriever: Retriever,
        store: JsonStore | None = None,
    ) -> None:
        """Initialize the search service.

        Args:
            retriever: Retriever used to find relevant indexed chunks.
            store: Optional storage backend used to load datasets and persist
                search results. If omitted, a new ``JsonStore`` is created.
        """
        self.retriever = retriever
        self.store = store or JsonStore()

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """Retrieve the top-k chunks for a query.

        Args:
            query: Search query used to retrieve relevant chunks.
            k: Maximum number of chunks to return.

        Returns:
            A list of the most relevant chunks ordered by their retrieval
            score.

        Raises:
            ValueError: If the query is empty or ``k`` is not greater than
                zero.
            RetrievalError: If the underlying index is invalid.
        """

        return self.retriever.search(
            query=query,
            k=k,
        )

    @staticmethod
    def result(
        question: str,
        chunks: list[Chunk],
        question_id: str | None = None,
    ) -> MinimalSearchResults:
        """Convert retrieved chunks into a public search result model.

        A new UUID is generated when no question identifier is provided.

        Args:
            question: Question or query associated with the search results.
            chunks: Chunks retrieved for the question.
            question_id: Optional identifier for the question. If omitted,
                a new UUID is generated.

        Returns:
            A ``MinimalSearchResults`` containing the question and its
            retrieved source references.
        """

        return MinimalSearchResults(
            question_id=question_id or str(uuid.uuid4()),
            question=question,
            retrieved_sources=[
                chunk.to_minimal_source()
                for chunk in chunks
            ],
        )

    def search_dataset(
        self,
        path: Path,
        k: int,
        output_dir: Path,
    ) -> None:
        """Search every question in a dataset and persist the results.

        The dataset is loaded from ``path``, each question is searched using
        the configured retriever, and the resulting source references are
        saved to the specified output directory.

        Args:
            path: Path to the dataset containing the questions to search.
            k: Maximum number of chunks to retrieve for each question.
            output_dir: Directory where the generated search results are
                saved.

        Raises:
            ValueError: If ``k`` is not greater than zero or a query is empty.
            RetrievalError: If the underlying index is invalid.
        """

        dataset = self.store.load_dataset(path)

        results: list[MinimalSearchResults] = []

        for question in tqdm(
            dataset.rag_questions,
            desc="Searching",
            unit="q",
        ):
            chunks = self.search(
                question.question,
                k,
            )

            results.append(
                self.result(
                    question=question.question,
                    chunks=chunks,
                    question_id=question.question_id,
                )
            )

        output = StudentSearchResults(
            search_results=results,
            k=k,
        )

        self.store.save(
            output,
            path,
            output_dir,
        )

    def search_one(
        self,
        query: str,
        k: int = 10,
    ) -> list[MinimalSource]:
        """Search for a query and return its source references.

        Args:
            query: Search query used to retrieve relevant chunks.
            k: Maximum number of chunks to retrieve.

        Returns:
            A list of minimal source references corresponding to the
            retrieved chunks.

        Raises:
            ValueError: If the query is empty or ``k`` is not greater than
                zero.
            RetrievalError: If the underlying index is invalid.
        """
        chunks = self.search(query, k)

        return [
            chunk.to_minimal_source()
            for chunk in chunks
        ]
