from pathlib import Path

from .config import (
    ANSWERED_DOCS_DATASET,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_MAX_CHUNK_SIZE,
    RESULTS_PATH,
    SEARCH_RESULTS_DIR,
    UNANSWERED_DOCS_DATASET,
    SEARCH_RESULTS_UNANSWERED_DOCS_DIR
)

from .generation import AnswerGenerator, AnswerService
from .indexing import Indexer, IndexCache, IndexStorage
from .retrieval import BM25Retriever, SearchService, HybridRetriever, SemanticRetriever, QueryCache
from .evaluation import Evaluator
from .models import MinimalSearchResults, MinimalSource, StudentSearchResults

import uuid
import time


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

        stats = indexer.index(max_chunk_size=max_chunk_size)

        print("\n=== Indexing statistics ===")
        print(f"Files indexed: {stats['files_indexed']}")
        print(f"Unchanged files: {stats['unchanged_files']}")
        print(f"Modified files: {stats['modified_files']}")
        print(f"New files: {stats['new_files']}")
        print(f"Deleted files: {stats['deleted_files']}")
        print(f"Total chunks: {stats['chunks']}")

        print(f"\nIngestion complete! Indices saved under {processed_dir}")

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
        self._print_search_results(query, k, sources)

    def search_dataset(
        self,
        dataset_path: str = str(UNANSWERED_DOCS_DATASET),
        k: int = 10,
        save_directory: str = str(SEARCH_RESULTS_UNANSWERED_DOCS_DIR),
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
        student_search_results_path: str = str(RESULTS_PATH),
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
        student_search_results_path: str = str(RESULTS_PATH),
        dataset_path: str = str(ANSWERED_DOCS_DATASET),
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
        """Build the lexical index and generate the semantic embedding matrix.

        Args:
            max_chunk_size (int): The maximum size allowed for each text chunk.
                Defaults to DEFAULT_MAX_CHUNK_SIZE.
            raw_dir (str): Path to the directory containing raw data.
                Defaults to str(DATA_RAW).
            processed_dir (str): Path to the directory where processed indices will be saved.
                Defaults to str(DATA_PROCESSED).

        Returns:
            None
        """
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
        """Retrieve the top-k sources using semantic similarity only.

        Args:
            query (str): The search text or question to look up.
            k (int): The number of top relevant sources to return. Defaults to 10.
            processed_dir (str): Path to the directory containing the processed indices.
                Defaults to str(DATA_PROCESSED).

        Returns:
            None
        """
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
        """Retrieve the top-k sources by fusing BM25 and semantic rankings.

        Args:
            query (str): The search text or question to look up.
            k (int): The number of top relevant sources to return. Defaults to 10.
            processed_dir (str): Path to the directory containing the processed indices.
                Defaults to str(DATA_PROCESSED).

        Returns:
            None
        """
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

    def api(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        processed_dir: str = str(DATA_PROCESSED),
    ) -> None:
        """Start the local HTTP API.

        Args:
            host (str): The network address to bind the API server to.
                Defaults to "127.0.0.1".
            port (int): The port network number to listen on. Defaults to 8000.
            processed_dir (str): Path to the directory containing the processed indices.
                Defaults to str(DATA_PROCESSED).

        Returns:
            None
        """
        from .api import create_app
        import uvicorn

        app = create_app(processed_dir)

        uvicorn.run(
            app,
            host=host,
            port=port,
        )

    @staticmethod
    def _print_search_results(
        query: str,
        k: int,
        sources: list[MinimalSource],
    ) -> None:
        """Format and print search results as an indented JSON string.

        Args:
            query (str): The search query that generated these results.
            k (int): The maximum number of sources requested.
            sources (list[MinimalSource]): A list of extracted source documents.

        Returns:
            None
        """
        output = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=str(uuid.uuid4()),
                    question=query,
                    retrieved_sources=sources,
                )
            ],
            k=k,
        )
        print(output.model_dump_json(indent=2))

    def caching(
        self,
        processed_dir: str = str(DATA_PROCESSED),
        query_cache_dir: str = str(DATA_PROCESSED.parent / "query_cache"),
        query: str = "what is rag",
        k: int = 5,
    ) -> None:
        """Demonstrate index and query cache performance.

        Args:
            processed_dir (str): Path to the directory containing the processed indices.
                Defaults to str(DATA_PROCESSED).
            query_cache_dir (str): Path to the directory where query cache is stored.
                Defaults to str(DATA_PROCESSED.parent / "query_cache").
            query (str): The mock query used for the caching demonstration.
                Defaults to "what is rag".
            k (int): The number of sources to request for the demo. Defaults to 5.

        Returns:
            None
        """

        # --- Index cache demo ---
        storage = IndexStorage(processed_dir)
        IndexCache.clear()

        t0 = time.perf_counter()
        idx1 = IndexCache.get(storage)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        idx2 = IndexCache.get(storage)
        t3 = time.perf_counter()

        print(f"Cold load:   {(t1 - t0):.5f} s")
        print(f"Cached load: {(t3 - t2):.7f} s")
        print(f"Same object in RAM (idx1 is idx2): {idx1 is idx2}")

        # --- Query cache demo ---
        query_cache = QueryCache(query_cache_dir)
        query_cache.clear()

        query = "what is rag"
        k = 5

        print(f"SHA-256 key: {query_cache.key_for(query, k)}")
        print(f"Contains before search: {query_cache.contains(query, k)}")

        retriever = BM25Retriever(processed_dir)
        service = SearchService(retriever, query_cache=query_cache)

        t4 = time.perf_counter()
        r1 = service.search(query, k=k)
        t5 = time.perf_counter()

        t6 = time.perf_counter()
        r2 = service.search(query, k=k)
        t7 = time.perf_counter()

        print(f"Cold query: {(t5 - t4):.7f} s")
        print(f"Cached query: {(t7 - t6):.7f} s")
        print(f"Query cache hit (r1 == r2): {r1 == r2}")
        print(f"Contains after search: {query_cache.contains(query, k)}")
