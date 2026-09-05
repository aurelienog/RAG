import uuid
from pathlib import Path

from tqdm import tqdm

from ..domain import Chunk
from ..ingest import JsonStore
from ..models import MinimalSearchResults, StudentSearchResults
from .bm25_retriever import Retriever


class SearchService:
    """Search indexed chunks and persist search results."""

    def __init__(
        self,
        retriever: Retriever,
        store: JsonStore | None = None,
    ) -> None:
        self.retriever = retriever
        self.store = store or JsonStore()

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Chunk]:
        """Return the top-k chunks for a query."""

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
        """Convert retrieved chunks into the public result model."""

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
        """Search every question in a dataset and save the results."""

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
    ) -> StudentSearchResults:
        """Search one query and return the public result model."""

        return StudentSearchResults(
            search_results=[
                self.result(
                    query,
                    self.search(query, k),
                )
            ],
            k=k,
        )