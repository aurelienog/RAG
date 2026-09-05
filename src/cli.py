from pathlib import Path

from .config import (
    DATA_RAW,
    DATA_PROCESSED,
    SEARCH_RESULTS_DIR,
    DEFAULT_MAX_CHUNK_SIZE,
)
from .generation import AnswerGenerator, AnswerService
from .indexing.indexer import Indexer
from .retrieval import Retriever, SearchService
from .evaluation import Evaluator


class CLI:
    """Command-line interface for the RAG system."""

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_dir: str = str(DATA_RAW),
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Build the search index from a source tree."""

        indexer = Indexer(
            raw_dir=Path(raw_dir),
            processed_dir=Path(processed_dir),
        )

        stats = indexer.index(max_chunk_size=max_chunk_size)

        print(
            f"Indexed {stats['files_indexed']} files "
            f"into {stats['chunks']} chunks."
        )

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Retrieve the top-k sources for a single query."""

        search = SearchService(
            Retriever(Path(processed_dir))
        )

        output = search.search_one(query, k)

        print(output.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = str(SEARCH_RESULTS_DIR),
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Run retrieval over a question dataset."""

        dataset_path_obj = Path(dataset_path)
        output_path = Path(save_directory)

        search = SearchService(
            Retriever(Path(processed_dir))
        )

        search.search_dataset(
            dataset_path_obj,
            k,
            output_path,
        )

        print(
            "Saved student_search_results to "
            f"{output_path / dataset_path_obj.name}"
        )

    def answer(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Generate an answer for a single query."""

        search = SearchService(
            Retriever(Path(processed_dir))
        )

        answer_service = AnswerService(
            search,
            AnswerGenerator(),
        )

        output = answer_service.answer(query, k)

        print(output.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        processed_dir: str,
    ) -> None:
        """Generate answers from previously generated search results."""

        results_path = Path(student_search_results_path)
        output_path = Path(save_directory)

        search = SearchService(
            Retriever(Path(processed_dir))
        )

        answer_service = AnswerService(
            search,
            AnswerGenerator(),
        )

        answer_service.answer_dataset(
            results_path,
            output_path,
        )

        print(
            "Saved student_search_results_and_answer to "
            f"{output_path / results_path.name}"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval recall@k."""

        evaluator = Evaluator()

        evaluator.evaluate(
            Path(student_search_results_path),
            Path(dataset_path),
        )
