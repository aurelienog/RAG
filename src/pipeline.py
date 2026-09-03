import json
from pathlib import Path
from typing import List
from tqdm import tqdm

from .domain import Chunk, DatasetError
from .retrieval.bm25_retriever import Retriever
from .generation.generator import AnswerGenerator

from .models import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    RagDataset
)


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation steps.
    Connects the Retriever with the AnswerGenerator and formats output datasets.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def search(self, query: str, k: int = 10) -> List[Chunk]:
        """
        Executes a single query search returning domain Chunk objects.
        Used by the single-query CLI command.
        """
        return self.retriever.search(query=query, k=k)

    def search_dataset(
        self,
        dataset_path: Path,
        k: int,
        save_directory: Path
    ) -> None:
        """
        Processes a JSON dataset of questions, executes retrieval for each,
        and exports a validated StudentSearchResults JSON file.
        """

        try:
            with dataset_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            dataset = RagDataset.model_validate(data)

        except (json.JSONDecodeError, OSError, Exception) as exc:
            raise DatasetError(f"Failed to read/validate dataset: {dataset_path}") from exc

        search_results_list = []

        for q in tqdm(dataset.rag_questions, desc="Searching dataset", unit="query"):
            hits = self.search(query=q.question, k=k)

            retrieved_sources = [
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.start,
                    last_character_index=chunk.end
                )
                for chunk in hits
            ]

            result = MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=retrieved_sources
            )

            search_results_list.append(result)

        output_payload = StudentSearchResults(
            search_results=search_results_list,
            k=k
        )

        save_directory.mkdir(parents=True, exist_ok=True)
        output_path = save_directory / dataset_path.name

        try:
            with output_path.open("w", encoding="utf-8") as f:
                f.write(output_payload.model_dump_json(indent=2))
        except OSError as exc:
            raise DatasetError(f"Failed to write search results: {exc}")

    def answer

    def answer_dataset