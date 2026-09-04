from pathlib import Path

from .config import (
    DATA_RAW,
    DATA_PROCESSED,
    SEARCH_RESULTS_DIR,
    DEFAULT_MAX_CHUNK_SIZE
)
from .generation import AnswerGenerator, AnswerService
from .indexing.indexer import Indexer
from .retrieval import Retriever, SearchService
from .evaluation import Evaluator


class CLI:
    """Command-line interface for the RAG system.

    Provides commands to ingest and index codebases, search relevant snippets,
    generate answers using local models, and evaluate retrieval metrics.
    """

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_dir: str = str(DATA_RAW),
        processed_dir: str = str(DATA_PROCESSED)
    ) -> None:
        """Ingest a source tree and build the search index under a processed dir.

        Args:
            raw_dir (str): Directory containing the source corpus to index.
            processed_dir (str): Directory where the index JSON will be written.
            max_chunk_size (int): Maximum character length allowed per chunk.
                Defaults to 2000.
        """
        if max_chunk_size <= 0 or max_chunk_size > DEFAULT_MAX_CHUNK_SIZE:
            raise ValueError("max_chunk_size must be between 1 and "
                             f"{DEFAULT_MAX_CHUNK_SIZE} characters.")

        source_dir = Path(raw_dir)
        if not source_dir.exists():
            raise ValueError(f"Input directory not found: {source_dir}")
        if not source_dir.is_dir():
            raise ValueError(f"Expected a directory: {source_dir}")

        processed_path = Path(processed_dir)
        if processed_path.exists() and not processed_path.is_dir():
            raise ValueError(f"Expected an output directory: {processed_path}")
        try:
            processed_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"Could not prepare output directory: {processed_path}"
            ) from exc

        output_dir = processed_path
        output_dir.mkdir(parents=True, exist_ok=True)

        self._indexer = Indexer(
            raw_dir=source_dir,
            processed_dir=processed_dir
        )
        stats = self._indexer.index(
            max_chunk_size=max_chunk_size,
        )
        print(f"Indexed {stats['files_indexed']} files into {stats['chunks']} chunks.")

    def search(
            self,
            query: str,
            k: int = 10,
            processed_dir: str = str(DATA_PROCESSED)
    ) -> None:
        """Retrieve the top-k source locations for a single query.

        Args:
            query (str): The natural language or code question to search.
            k (int): Number of most relevant sources to return. Defaults to 10.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        index_path = Path(processed_dir)
        if not index_path.exists():
            raise ValueError(f"Processed directory not found: {index_path}")
        if not index_path.is_dir():
            raise ValueError(f"Expected a directory: {index_path}")

        service = SearchService(Retriever(index_path))
        output = service.search_one(query, k)
        print(output.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = str(SEARCH_RESULTS_DIR),
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Run search over an entire question dataset and save the outputs.

        Args:
            dataset_path (str): Path to the input dataset JSON file.
            k (int): Number of sources to retrieve per question. Defaults to 10
            save_directory (str): Directory where the StudentSearchResults JSON
                file will be stored. Defaults to "data/output/search_results".
        """

        dataset_path_obj = Path(dataset_path)
        output_path = Path(save_directory)
        index_path = Path(processed_dir)

        if not dataset_path_obj.exists():
            raise ValueError(f"Dataset not found: {dataset_path_obj}")
        if not dataset_path_obj.is_file():
            raise ValueError(f"Expected a file: {dataset_path_obj}")
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Expected an output directory: {output_path}")
        if not index_path.exists():
            raise ValueError(f"Processed directory not found: {index_path}")
        if not index_path.is_dir():
            raise ValueError(f"Expected a directory: {index_path}")
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        service = SearchService(Retriever(index_path))
        service.search_dataset(
            dataset_path_obj,
            k,
            output_path,
        )
        print(f"Saved student_search_results to {output_path / dataset_path_obj.name}")

    def answer(
            self,
            query: str,
            k: int = 10,
            processed_dir: str = str(DATA_PROCESSED)
    ) -> None:
        """Answer a single query using the retrieved context window.

        Args:
            query (str): The question to be answered by the model.
            k (int): Number of context sources to feed into the generator.
                Defaults to 10.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        index_path = Path(processed_dir)
        if not index_path.exists():
            raise ValueError(f"Processed directory not found: {index_path}")
        if not index_path.is_dir():
            raise ValueError(f"Expected a directory: {index_path}")

        search = SearchService(Retriever(index_path))
        output = AnswerService(search, AnswerGenerator()).answer(query, k)
        print(output.model_dump_json(indent=2))
        # work in progress, debe retornar:
        # output = StudentSearchResultsAndAnswer(
        #     search_results=[answer],
        #     k=k,
        # )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        processed_dir: str,
    ) -> None:
        """Generate model answers for an entire dataset
        from its search results.

        Args:
            student_search_results_path (str): Path to the JSON containing
            previously generated search results.
            save_directory (str): Directory where the final
                StudentSearchResultsAndAnswer JSON file will be written.
        """
        results_path = Path(student_search_results_path)
        if not results_path.exists():
            raise ValueError(f"Search results not found: {results_path}")
        if not results_path.is_file():
            raise ValueError(f"Expected a file: {results_path}")
        output_path = Path(save_directory)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"Could not prepare output directory: {output_path}"
            ) from exc
        index_path = Path(processed_dir)
        if not index_path.exists():
            raise ValueError(f"Processed directory not found: {index_path}")
        if not index_path.is_dir():
            raise ValueError(f"Expected a directory: {index_path}")

        search = SearchService(Retriever(index_path))
        AnswerService(search, AnswerGenerator()).answer_dataset(
            results_path,
            output_path,
        )
        print("Saved student_search_results_and_answer to "
              f"{output_path / results_path.name}")

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
        results_path = Path(student_search_results_path)
        dataset_path_obj = Path(dataset_path)
        if not results_path.exists():
            raise ValueError(f"Search results not found: {results_path}")
        if not results_path.is_file():
            raise ValueError(f"Expected a file: {results_path}")
        if not dataset_path_obj.exists():
            raise ValueError(f"Dataset not found: {dataset_path_obj}")
        if not dataset_path_obj.is_file():
            raise ValueError(f"Expected a file: {dataset_path_obj}")

        evaluator = Evaluator()
        evaluator.evaluate(results_path, dataset_path=dataset_path_obj)
