import pytest

from src.cli import CLI
from src.domain import DatasetError
from src.generation import build_prompt
from src.ingest import JsonStore
from src.retrieval import Retriever


def test_retriever_rejects_invalid_input() -> None:
    retriever = Retriever.__new__(Retriever)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("", k=10)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("   ", k=10)

    with pytest.raises(ValueError, match="k must be greater than 0"):
        retriever.search("query", k=0)

    with pytest.raises(ValueError, match="k must be greater than 0"):
        retriever.search("query", k=-1)


def test_cli_search_handles_degenerate_input(monkeypatch, capsys) -> None:
    class FakeRetriever:
        def __init__(self, processed_dir) -> None:
            pass

        def search(self, query: str, k: int) -> list:
            return []

    class FakeSearchService:
        def __init__(self, retriever) -> None:
            self.retriever = retriever

        def search_one(self, query: str, k: int):
            return []

    monkeypatch.setattr(
        "src.cli.SearchService",
        FakeSearchService,
    )
    monkeypatch.setattr(
        "src.cli.Retriever",
        FakeRetriever,
    )

    CLI().search("", k=0)

    assert capsys.readouterr().out == ""


def test_json_store_rejects_corrupt_json(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(DatasetError, match="Invalid dataset format"):
        JsonStore().load_dataset(dataset_path)


def test_json_store_reports_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(DatasetError, match="Dataset not found"):
        JsonStore().load_dataset(missing_path)


def test_json_store_reports_missing_search_results(tmp_path) -> None:
    missing_path = tmp_path / "search_results.json"

    with pytest.raises(DatasetError, match="Dataset not found"):
        JsonStore().load_search_results(missing_path)


def test_generation_prompt_contains_question_and_context() -> None:
    prompt = build_prompt(
        question="How is the index loaded?",
        context="IndexStorage.load reads index.json.",
    )

    assert "How is the index loaded?" in prompt
    assert "IndexStorage.load reads index.json." in prompt
    assert "Do not invent facts" in prompt
