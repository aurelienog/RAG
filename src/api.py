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
    """Request body for the RAG query endpoint."""

    question: str = Field(min_length=1)
    k: int = Field(default=10, gt=0)


class QueryResponse(BaseModel):
    """Response returned by the RAG query endpoint."""

    question: str
    answer: str
    sources: list[MinimalSource]


def create_app(
    processed_dir: str | Path = DATA_PROCESSED,
) -> FastAPI:
    """Create the local RAG HTTP API."""

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
        """Redirect the root URL to the interactive API documentation."""
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
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
