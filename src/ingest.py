import json
from pathlib import Path

from .domain import Chunk, DatasetError
from .models import (
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class JsonStore:
    """Load and save the JSON contracts exchanged by the pipeline."""

    def load_dataset(self, path: Path) -> RagDataset:
        """Load and validate a question dataset."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RagDataset.model_validate(data)
        except FileNotFoundError as exc:
            raise DatasetError(f"Dataset not found: {path}") from exc
        except OSError as exc:
            raise DatasetError(f"Could not read dataset: {path}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise DatasetError(f"Invalid dataset format: {path}") from exc

    def load_search_results(self, path: Path) -> StudentSearchResults:
        """Load and validate persisted search results."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return StudentSearchResults.model_validate(data)
        except FileNotFoundError as exc:
            raise DatasetError(f"Dataset not found: {path}") from exc
        except OSError as exc:
            raise DatasetError(f"Could not read dataset: {path}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise DatasetError(f"Invalid dataset format: {path}") from exc

    def save(
        self,
        output: StudentSearchResults | StudentSearchResultsAndAnswer,
        source_path: Path,
        directory: Path,
    ) -> None:
        """Save a validated output model using the input filename."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            output_path = directory / source_path.name
            output_path.write_text(
                output.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise DatasetError(
                f"Could not write output: {directory / source_path.name}"
            ) from exc

class SourceResolver:
    """Resolve persisted source locations against indexed chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = {
            (
                chunk.file_path,
                chunk.start,
                chunk.end,
            ): chunk
            for chunk in chunks
        }

    def resolve(self, sources: list[MinimalSource]) -> list[Chunk]:
        """Return chunks matching the supplied source locations."""
        chunks: list[Chunk] = []

        for source in sources:
            key = (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )

            chunk = self._chunks.get(key)

            if chunk is None:
                raise DatasetError(
                    f"Source not found in index: {source.file_path} "
                    f"[{source.first_character_index}, "
                    f"{source.last_character_index}]"
                )

            chunks.append(chunk)

        return chunks


def build_context(
    chunks: list[Chunk],
    max_characters: int = 12_000,
) -> str:
    """Build a bounded context from retrieved chunks."""

    if max_characters <= 0:
        return ""

    sections: list[str] = []
    used = 0

    for chunk in chunks:
        section = f"Source: {chunk.file_path}\n{chunk.text}"
        separator_length = 2 if sections else 0
        required = separator_length + len(section)

        if used + required > max_characters:
            break

        sections.append(section)
        used += required

    return "\n\n".join(sections)
