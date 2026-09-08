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
    """Load and save JSON contracts exchanged by the RAG pipeline."""

    def load_dataset(self, path: Path) -> RagDataset:
        """Load and validate a RAG question dataset from a JSON file.

        Args:
            path: Path to the JSON dataset file.

        Returns:
            A validated ``RagDataset`` containing the dataset questions.

        Raises:
            DatasetError: If the file does not exist, cannot be read, contains
                invalid JSON, or does not match the expected dataset schema.
        """

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
        """Load and validate persisted search results from a JSON file.

        Args:
            path: Path to the JSON file containing search results.

        Returns:
            A validated ``StudentSearchResults`` instance.

        Raises:
            DatasetError: If the file does not exist, cannot be read, contains
                invalid JSON, or does not match the expected search results
                schema.
        """

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
        """Save a validated pipeline output as a JSON file.

        The output file uses the same filename as ``source_path`` and is
        written to ``directory``. The destination directory is created if it
        does not already exist.

        Args:
            output: Validated search results or search results with generated
                answers to persist.
            source_path: Path whose filename is used for the output file.
            directory: Directory where the output JSON file is saved.

        Raises:
            DatasetError: If the output directory cannot be created or the
                output file cannot be written.
        """

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
        """Initialize the source resolver with indexed chunks.

        Args:
            chunks: Chunks available in the persisted search index. Each chunk
                is indexed by its file path and character range.
        """
        self._chunks = {
            (
                chunk.file_path,
                chunk.start,
                chunk.end,
            ): chunk
            for chunk in chunks
        }

    def resolve(self, sources: list[MinimalSource]) -> list[Chunk]:
        """Resolve source references to their corresponding indexed chunks.

        Args:
            sources: Source references containing file paths and character
                ranges.

        Returns:
            A list of indexed chunks corresponding to the supplied source
            references, in the same order.

        Raises:
            DatasetError: If any source reference cannot be found in the
                indexed chunks.
        """
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
    """Build a bounded text context from retrieved chunks.

    Chunks are added in the order provided until adding the next chunk would
    exceed ``max_characters``. Each chunk is prefixed with its source file
    path, and consecutive sections are separated by a blank line.

    Args:
        chunks: Retrieved chunks to include in the context.
        max_characters: Maximum number of characters allowed in the generated
            context.

    Returns:
        A formatted context containing the selected source chunks. An empty
        string is returned when ``max_characters`` is not greater than zero or
        when no chunks fit within the limit.
    """

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
