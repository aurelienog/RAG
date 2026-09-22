from pathlib import Path

from .config import (
    DATA_RAW,
    DATA_PROCESSED,
    SEARCH_RESULTS_DIR,
    DEFAULT_MAX_CHUNK_SIZE,
)
from .generation import AnswerGenerator, AnswerService
from .indexing import Indexer
from .retrieval import BM25Retriever, SearchService, HybridRetriever, SemanticRetriever
from .evaluation import Evaluator


class CLI:
    """Provide command-line operations for the RAG system."""

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_dir: str = str(DATA_RAW),
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Build and persist the search index from a source tree.

        Args:
            max_chunk_size: Maximum number of characters allowed in each
                generated chunk.
            raw_dir: Directory containing the source files to index.
            processed_dir: Directory where the generated index is stored.
        """

        indexer = Indexer(
            raw_dir=Path(raw_dir),
            processed_dir=Path(processed_dir),
        )

        indexer.index(max_chunk_size=max_chunk_size)
        print(f"Ingestion complete! Indices saved under {processed_dir}")

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Retrieve and display the top-k sources for a single query.

        Args:
            query: Search query used to retrieve relevant sources.
            k: Maximum number of sources to retrieve.
            processed_dir: Directory containing the persisted search index.
        """

        search = SearchService(
            BM25Retriever(Path(processed_dir))
        )

        sources = search.search_one(query, k)
        for source in sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}, "
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = str(SEARCH_RESULTS_DIR),
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Run retrieval for every question in a dataset.

        The generated search results are persisted to the configured output
        directory.

        Args:
            dataset_path: Path to the dataset containing the questions.
            k: Maximum number of sources to retrieve for each question.
            save_directory: Directory where the search results are saved.
            processed_dir: Directory containing the persisted search index.
        """

        dataset_path_obj = Path(dataset_path)
        output_path = Path(save_directory)

        search = SearchService(
            BM25Retriever(Path(processed_dir))
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
        """Generate and display an answer for a single query.

        Args:
            query: Question to answer using the indexed source documents.
            k: Number of relevant sources to retrieve as context.
            processed_dir: Directory containing the persisted search index.
        """

        search = SearchService(
            BM25Retriever(Path(processed_dir))
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
        save_directory: str = str(SEARCH_RESULTS_DIR),
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Generate answers from previously generated search results.

        The generated answers are persisted to the configured output
        directory.

        Args:
            student_search_results_path: Path to the file containing the
                previously generated search results.
            save_directory: Directory where the generated answers are saved.
            processed_dir: Directory containing the persisted search index.
        """

        results_path = Path(student_search_results_path)
        output_path = Path(save_directory)

        search = SearchService(
            BM25Retriever(Path(processed_dir))
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
        """Evaluate retrieval performance using recall@k.

        Args:
            student_search_results_path: Path to the generated student search
                results to evaluate.
            dataset_path: Path to the reference dataset containing the expected
                sources.
        """

        evaluator = Evaluator()

        evaluator.evaluate(
            Path(student_search_results_path),
            Path(dataset_path),
        )

    def index_semantic(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_dir: str = str(DATA_RAW),
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Build the lexical index and generate the semantic embedding matrix."""
        indexer = Indexer(
            raw_dir=Path(raw_dir),
            processed_dir=Path(processed_dir),
        )

        indexer.index(
            max_chunk_size=max_chunk_size,
            build_semantic_embeddings=True,
        )
        print(
            "Ingestion complete! Lexical + semantic indices saved under "
            f"{processed_dir}"
        )

    def search_semantic(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Retrieve the top-k sources using semantic similarity only."""
        search = SearchService(
            SemanticRetriever(Path(processed_dir))
        )

        sources = search.search_one(query, k)
        for source in sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}, "
                f"{source.last_character_index}]"
            )

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Retrieve the top-k sources by fusing BM25 and semantic rankings."""
        search = SearchService(
            HybridRetriever(
                bm25_retriever=BM25Retriever(Path(processed_dir)),
                semantic_retriever=SemanticRetriever(Path(processed_dir)),
            )
        )

        sources = search.search_one(query, k)
        for source in sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}, "
                f"{source.last_character_index}]"
            )
