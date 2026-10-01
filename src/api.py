from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .config import DATA_PROCESSED
from .models import MinimalSource
from .domain import RAGError
from .generation import AnswerGenerator, AnswerService
from .retrieval import BM25Retriever, SearchService


class QueryRequest(BaseModel):
    """Request body for the RAG query endpoint.

    Attributes:
        question (str): The search text or query question. Must not be empty.
        k (int): The number of top relevant sources to retrieve. Must be greater than 0.
            Defaults to 10.
    """

    question: str = Field(min_length=1)
    k: int = Field(default=10, gt=0)


class QueryResponse(BaseModel):
    """Response returned by the RAG query endpoint.

    Attributes:
        question (str): The original question that was asked.
        answer (str): The generated answer from the RAG service.
        sources (list[MinimalSource]): List of text sources used to generate the answer.
    """

    question: str
    answer: str
    sources: list[MinimalSource]


def create_app(
    processed_dir: str | Path = DATA_PROCESSED,
) -> FastAPI:
    """Create the local RAG HTTP API.

    Initializes the FastAPI application instance, configures the internal BM25 search
    and answer generation services using the specified database directory, and registers
    all routing endpoints.

    Args:
        processed_dir (str | Path): Path to the directory where the processed indices
            are located. Defaults to DATA_PROCESSED.

    Returns:
        FastAPI: A fully configured FastAPI application instance.
    """

    app = FastAPI(
        title="RAG against the machine",
        version="1.0.0",
    )

    search = SearchService(
        BM25Retriever(Path(processed_dir))
    )

    answer_service = AnswerService(
        search,
        AnswerGenerator(),
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Redirect the root URL to the interactive API documentation.

        Returns:
            RedirectResponse: A redirection response pointing to the '/docs' path.
        """
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Check the operational availability of the service.

        Returns:
            dict[str, str]: A dictionary indicating the current operational status.
                Example: {"status": "ok"}
        """
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        """Process a text query, retrieve relevant sources, and generate an answer.

        Args:
            request (QueryRequest): The incoming request payload containing the
                question text and retrieval limits.

        Returns:
            QueryResponse: The payload containing the validated question, the final
                generated answer, and the metadata of the retrieved source documents.

        Raises:
            HTTPException: 400 status error if argument parameters are invalid,
                or 500 status error if an internal system RAG engine exception occurs.
        """
        try:
            result = answer_service.answer(
                request.question,
                request.k,
            )

            answer = result.search_results[0]

            return QueryResponse(
                question=answer.question,
                answer=answer.answer,
                sources=answer.retrieved_sources,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except RAGError as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    return app


app = create_app()
