import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .domain import Chunk, DatasetError
from .models import (
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


Model = TypeVar("Model", bound=BaseModel)


class JsonStore:
    """Load and save the JSON contracts exchanged by the pipeline."""

    def load_dataset(self, path: Path) -> RagDataset:
        """Load and validate a question dataset."""
        return self._load(path, RagDataset, "Dataset")

    def load_search_results(self, path: Path) -> StudentSearchResults:
        """Load and validate persisted search results."""
        return self._load(path, StudentSearchResults, "Search results")

    def save(
        self,
        output: StudentSearchResults | StudentSearchResultsAndAnswer,
        source_path: Path,
        directory: Path,
    ) -> None:
        """Save a validated output model using the input filename."""
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / source_path.name
        try:
            output_path.write_text(
                output.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise DatasetError(f"Could not write output: {output_path}") from exc

    @staticmethod
    def _load(path: Path, model: type[Model], name: str) -> Model:
        """Load JSON and validate it with a Pydantic model."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return model.model_validate(data)
        except FileNotFoundError as exc:
            raise DatasetError(f"{name} not found: {path}") from exc
        except OSError as exc:
            raise DatasetError(f"Could not read {name.lower()}: {path}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise DatasetError(f"Invalid {name.lower()} format: {path}") from exc


class SourceResolver:
    """Resolve persisted source locations against indexed chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = {
            (chunk.file_path, chunk.start, chunk.end): chunk
            for chunk in chunks
        }

    def resolve(self, sources: list[MinimalSource]) -> list[Chunk]:
        """Return chunks matching the supplied source locations."""
        return [
            self._chunks[key]
            for source in sources
            if (key := (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )) in self._chunks
        ]


def build_context(chunks: list[Chunk], max_characters: int = 12_000) -> str:
    """Format retrieved chunks into bounded model context."""
    if max_characters <= 0:
        return ""

    sections: list[str] = []
    used = 0
    for chunk in chunks:
        section = f"Source: {chunk.file_path}\n{chunk.text}"
        separator = 2 if sections else 0
        available = max_characters - used - separator
        if available <= 0:
            break
        sections.append(section[:available])
        used += separator + min(len(section), available)
        if len(section) > available:
            break
    return "\n\n".join(sections)
