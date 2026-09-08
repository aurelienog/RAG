from pydantic import BaseModel, Field

import uuid


class MinimalSource(BaseModel):
    """Represent a source location within an indexed file.

    Attributes:
        file_path: Path of the source file.
        first_character_index: Inclusive character offset where the source
            begins.
        last_character_index: Exclusive character offset where the source
            ends.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question without a ground-truth answer.

    Attributes:
        question_id: Unique identifier for the question. A UUID is generated
            automatically when no identifier is provided.
        question: Text of the question.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its expected sources and answer.

    Attributes:
        sources: Ground-truth source locations supporting the answer.
        answer: Expected answer to the question.
    """
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset containing answered and unanswered questions.

    Attributes:
        rag_questions: Questions included in the RAG evaluation dataset.
            Each question may be either answered or unanswered.
    """
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent the sources retrieved for a single question.

    Attributes:
        question_id: Unique identifier of the searched question.
        question: Text of the question used for retrieval.
        retrieved_sources: Source locations returned by the retrieval
            system.
    """
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieval results together with a generated answer.

    Attributes:
        answer: Answer generated from the retrieved source context.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """Represent retrieval results for a complete question dataset.

    Attributes:
        search_results: Retrieval results for each searched question.
        k: Maximum number of sources requested for each question.
    """
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent retrieval results and generated answers for a dataset.

    Attributes:
        search_results: Retrieval results and generated answers for each
            question.
        k: Maximum number of sources retrieved for each question.
    """
    search_results: list[MinimalAnswer]
    k: int
