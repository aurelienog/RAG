from pathlib import Path

from .generation.generator import AnswerGenerator
from .indexing.indexer import Indexer
from .pipeline import RAGPipeline
from .retrieval.bm25_retriever import Retriever
from .evaluation.evaluator import Evaluator


class CLI:
    """Command-line interface for the RAG system.

    Provides commands to ingest and index codebases, search relevant snippets,
    generate answers using local models, and evaluate retrieval metrics.
    """

    def __init__(self) -> None:
        """Initialize the RAG services."""
        self._indexer = Indexer()
        self._retriever = Retriever()
        self._generator = AnswerGenerator()
        self._pipeline = RAGPipeline(
            retriever=self._retriever,
            generator=self._generator,
        )
        self._evaluator = Evaluator()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the search index under data/processed/.

        Args:
            max_chunk_size (int): Maximum character length allowed per chunk.
                Defaults to 2000.
        """
        self._indexer.index(
            max_chunk_size=max_chunk_size,
        )

    def search(self, query: str, k: int = 10) -> None:
        """Retrieve the top-k source locations for a single query.

        Args:
            query (str): The natural language or code question to search.
            k (int): Number of most relevant sources to return. Defaults to 10.
        """
        results = self._pipeline.search(query, k)
        print(results)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Run search over an entire question dataset and save the outputs.

        Args:
            dataset_path (str): Path to the input dataset JSON file.
            k (int): Number of sources to retrieve per question. Defaults to 10
            save_directory (str): Directory where the StudentSearchResults JSON
                file will be stored. Defaults to "data/output/search_results".
        """

        self._pipeline.search_dataset(
            dataset_path=Path(dataset_path),
            k=k,
            save_directory=Path(save_directory),
        )

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single query using the retrieved context window.

        Args:
            query (str): The question to be answered by the model.
            k (int): Number of context sources to feed into the generator.
                Defaults to 10.
        """
        answer = self._pipeline.answer(
            query=query,
            k=k,
        )
        print(answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate model answers for an entire dataset
        from its search results.

        Args:
            student_search_results_path (str): Path to the JSON containing
            previously generated search results.
            save_directory (str): Directory where the final
                StudentSearchResultsAndAnswer JSON file will be written.
        """
        self._pipeline.answer_dataset(
            student_search_results_path=Path(
                student_search_results_path
            ),
            save_directory=Path(save_directory),
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate local retrieval recall@k against a ground-truth dataset.

        Args:
            student_search_results_path (str): Path to your generated search
            results JSON file.
            dataset_path (str): Path to the ground-truth AnsweredQuestions
                dataset JSON file.
        """
        self._evaluator.evaluate(
            student_search_results_path=Path(
                student_search_results_path
            ),
            dataset_path=Path(dataset_path),
        )
