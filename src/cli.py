from pathlib import Path

from .config import (
    DATA_RAW,
    DATA_PROCESSED,
    SEARCH_RESULTS_DIR,
    DEFAULT_MAX_CHUNK_SIZE
)
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
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            raise ValueError("max_chunk_size must be between 1 and 2000 characters.")

        source_dir = Path(raw_dir)
        if not source_dir.exists():
            raise ValueError(f"Input directory not found: {source_dir}")
        if not source_dir.is_dir():
            raise ValueError(f"Expected a directory: {source_dir}")

        output_dir = Path(processed_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._indexer = Indexer(
            raw_dir=source_dir,
            processed_dir=processed_dir
        )
        indexed = self._indexer.index(
            max_chunk_size=max_chunk_size,
        )
        print(f"Indexed {indexed['files_indexed']} files into {indexed['chunks']} chunks.")
        print(f"Ingestion complete! Indices saved under {output_dir}")

    def search(
            self,
            query: str, k: int = 10,
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

        retriever = Retriever(processed_dir=processed_dir)
        pipeline = RAGPipeline(retriever=retriever, generator=AnswerGenerator())

        results = pipeline.search(query, k)
        print(results)

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
        save_directory_obj = Path(save_directory)

        if not dataset_path_obj.exists():
            raise ValueError(f"Input file not found: {dataset_path_obj}")
        if not dataset_path_obj.is_file():
            raise ValueError(f"Expected a file: {dataset_path_obj}")
        if save_directory_obj.exists() and save_directory_obj.is_file():
            raise ValueError(f"Output path cannot be a file: {save_directory_obj}")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        retriever = Retriever(processed_dir=processed_dir)
        pipeline = RAGPipeline(retriever=retriever, generator=AnswerGenerator())

        pipeline.search_dataset(
            dataset_path=dataset_path_obj,
            k=k,
            save_directory=save_directory_obj,
        )
        print(f"Saved student_search_results to {save_directory_obj / dataset_path_obj.name}")

    def answer(self, query: str, k: int = 10, processed_dir: str = str(DATA_PROCESSED)) -> None:
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

        retriever = Retriever(processed_dir=processed_dir)
        pipeline = RAGPipeline(retriever=retriever, generator=AnswerGenerator())

        answer = pipeline.answer(query=query, k=k)
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
        student_search_results_path_obj = Path(student_search_results_path)
        save_directory_obj = Path(save_directory)

        if not student_search_results_path_obj.exists():
            raise ValueError(f"Input file not found: {student_search_results_path_obj}")
        if not student_search_results_path_obj.is_file():
            raise ValueError(f"Expected a file: {student_search_results_path_obj}")
        if save_directory_obj.exists() and save_directory_obj.is_file():
            raise ValueError(f"Output path cannot be a file: {save_directory_obj}")

        save_directory_obj.mkdir(parents=True, exist_ok=True)

        pipeline = RAGPipeline(retriever=Retriever(), generator=AnswerGenerator())
        pipeline.answer_dataset(
            student_search_results_path=student_search_results_path_obj,
            save_directory=save_directory_obj,
        )
        print("Saved student_search_results_and_answer to "
              f"{save_directory_obj / student_search_results_path_obj}")

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
        student_search_results_path_obj = Path(student_search_results_path)
        dataset_path_obj = Path(dataset_path)

        if not student_search_results_path_obj.exists():
            raise ValueError(f"Input file not found: {student_search_results_path_obj}")
        if not student_search_results_path_obj.is_file():
            raise ValueError(f"Expected a file: {student_search_results_path_obj}")
        if not dataset_path_obj.exists():
            raise ValueError(f"Input file not found: {dataset_path_obj}")
        if not dataset_path_obj.is_file():
            raise ValueError(f"Expected a file: {dataset_path_obj}")

        evaluator = Evaluator()
        evaluator.evaluate(
            student_search_results_path=student_search_results_path_obj,
            dataset_path=dataset_path_obj,
        )
