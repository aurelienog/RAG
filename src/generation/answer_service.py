import uuid
from pathlib import Path

from tqdm import tqdm

from ..ingest import JsonStore
from .generator import AnswerGenerator
from ..domain import Chunk, DatasetError
from ..models import MinimalAnswer, StudentSearchResultsAndAnswer
from ..retrieval import SearchService
from ..ingest import SourceResolver, build_context


class AnswerService:
    """Generate grounded answers from retrieved chunks."""

    def __init__(
        self,
        search: SearchService,
        generator: AnswerGenerator,
        store: JsonStore | None = None,
    ) -> None:
        self.search = search
        self.generator = generator
        self.store = store or search.store
        self.resolve = SourceResolver(search.retriever.chunks)

    def answer(self, question: str, k: int = 10) -> MinimalAnswer:
        """Retrieve context and generate one structured answer."""
        chunks = self.search.search(question, k)
        return MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=question,
            retrieved_sources=[chunk.to_minimal_source() for chunk in chunks],
            answer=self._generate(question, chunks),
        )

    def answer_dataset(self, path: Path, output_dir: Path) -> None:
        """Generate answers for persisted search results."""
        search_results = self.store.load_search_results(path)
        answers: list[MinimalAnswer] = []
        for result in tqdm(search_results.search_results, desc="Answering", unit="q"):
            chunks = self.resolve.resolve(result.retrieved_sources)
            try:
                answer = self._generate(result.question, chunks)
            except DatasetError as exc:
                raise DatasetError(
                    f"Failed to answer question {result.question_id}."
                ) from exc
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )
        self.store.save(
            StudentSearchResultsAndAnswer(search_results=answers, k=search_results.k),
            path,
            output_dir,
        )

    def _generate(self, question: str, chunks: list[Chunk]) -> str:
        """Generate an answer from bounded retrieved context."""
        if not chunks:
            return "I could not find relevant information in the retrieved documents."
        try:
            return self.generator.generate(question, build_context(chunks))
        except Exception as exc:
            raise DatasetError("Failed to generate answer.") from exc
