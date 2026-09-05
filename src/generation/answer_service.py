import uuid
from pathlib import Path

from tqdm import tqdm

from ..ingest import JsonStore, SourceResolver, build_context
from ..models import MinimalAnswer, StudentSearchResultsAndAnswer
from ..retrieval import SearchService
from .generator import AnswerGenerator
from ..domain import Chunk


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
        self.resolver = SourceResolver(search.retriever.chunks)

    def answer(
        self,
        question: str,
        k: int = 10,
    ) -> StudentSearchResultsAndAnswer:
        """Retrieve context and generate one structured answer."""

        chunks = self.search.search(question, k)

        answer = MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=question,
            retrieved_sources=[
                chunk.to_minimal_source()
                for chunk in chunks
            ],
            answer=self._generate(question, chunks),
        )

        return StudentSearchResultsAndAnswer(
            search_results=[answer],
            k=k,
        )

    def answer_dataset(
        self,
        path: Path,
        output_dir: Path,
    ) -> None:
        """Generate answers for persisted search results."""

        search_results = self.store.load_search_results(path)

        answers: list[MinimalAnswer] = []

        for result in tqdm(
            search_results.search_results,
            desc="Answering",
            unit="q",
        ):
            chunks = self.resolver.resolve(
                result.retrieved_sources
            )

            answer = self._generate(
                result.question,
                chunks,
            )

            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k,
        )

        self.store.save(
            output,
            path,
            output_dir,
        )

    def _generate(
        self,
        question: str,
        chunks: list[Chunk],
    ) -> str:
        """Generate an answer from retrieved context."""

        if not chunks:
            return (
                "I could not find relevant information "
                "in the retrieved documents."
            )

        context = build_context(chunks)

        return self.generator.generate(
            question,
            context,
        )
