import uuid
from pathlib import Path

from tqdm import tqdm

from ..ingest import JsonStore, SourceResolver, build_context
from ..models import MinimalAnswer, StudentSearchResultsAndAnswer
from ..retrieval import SearchService
from .generator import AnswerGenerator
from ..domain import Chunk


class AnswerService:
    """Generate grounded answers from retrieved document chunks.

    This service retrieves relevant chunks for a question and uses an
    ``AnswerGenerator`` to produce an answer grounded in the retrieved
    context.

    Attributes:
        search: Service used to retrieve relevant chunks.
        generator: Component responsible for generating answers.
        store: Storage backend used to load and save search results.
        resolver: Resolver used to reconstruct chunks from source references.
    """

    def __init__(
        self,
        search: SearchService,
        generator: AnswerGenerator,
        store: JsonStore | None = None,
    ) -> None:
        """Initialize the answer service.

        Args:
            search: Service used to retrieve relevant document chunks.
            generator: Component used to generate answers from retrieved
                context.
            store: Optional storage backend used to load and save search
                results. If omitted, the store associated with ``search`` is
                used.
        """
        self.search = search
        self.generator = generator
        self.store = store or search.store
        self.resolver = SourceResolver(search.retriever.chunks)

    def answer(
        self,
        question: str,
        k: int = 10,
    ) -> StudentSearchResultsAndAnswer:
        """Retrieve context and generate a structured answer.

        Args:
            question: Question to answer.
            k: Maximum number of relevant chunks to retrieve.

        Returns:
            A structured result containing the generated answer and the
            sources used as retrieved context.
        """

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
        """Generate and persist answers for a set of search results.

        The existing search results are loaded from ``path``. Each retrieved
        source is resolved back to its corresponding chunk, and an answer is
        generated from the resulting context. The completed results are then
        saved to ``output_dir``.

        Args:
            path: Path to the persisted search results.
            output_dir: Directory where the generated answers are saved.
        """

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
        """Generate an answer using the retrieved chunks as context.

        Args:
            question: Question to answer.
            chunks: Retrieved document chunks used to build the answer
                context.

        Returns:
            The generated answer as a string. If no chunks are provided,
            returns a message indicating that no relevant information was
            found.
        """

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
