import json
from pathlib import Path

from ..domain import Chunk, IndexingError
from .lexical_index import LexicalIndex


class IndexStorage:
    """
    Persist and load the processed RAG index.
    """

    def __init__(self, processed_dir: str | Path) -> None:
        self.processed_dir = Path(processed_dir)

    def save(
        self,
        chunks: list[Chunk],
        lexical_index: LexicalIndex,
    ) -> None:
        """
        Save chunks and lexical index to JSON.
        """
        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "chunks": [
                {
                    "id": chunk.id,
                    "file_path": chunk.file_path,
                    "text": chunk.text,
                    "first_character_index": chunk.start,
                    "last_character_index": chunk.end,
                    "kind": chunk.kind,
                }
                for chunk in chunks
            ],
            "inverted_index": lexical_index.inverted_index,
            "doc_freq": lexical_index.doc_freq,
            "doc_lengths": lexical_index.doc_lengths,
            "avg_doc_length": lexical_index.avg_doc_length,
        }

        output_path = self.processed_dir / "index.json"

        try:
            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                )
        except OSError as exc:
            raise IndexingError(
                f"Could not write index: {output_path}"
            ) from exc

    def load(self) -> tuple[list[Chunk], LexicalIndex]:
        """
        Load chunks and lexical index from JSON.
        """
        input_path = self.processed_dir / "index.json"

        if not input_path.exists():
            raise IndexingError(
                f"Index file not found: {input_path}"
            )

        try:
            with input_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexingError(
                f"Could not read index: {input_path}"
            ) from exc

        try:
            chunks = [
                Chunk(
                    id=item["id"],
                    file_path=item["file_path"],
                    text=item["text"],
                    start=item["first_character_index"],
                    end=item["last_character_index"],
                    kind=item["kind"],
                )
                for item in payload["chunks"]
            ]

            lexical_index = LexicalIndex(
                inverted_index=payload["inverted_index"],
                doc_freq=payload["doc_freq"],
                doc_lengths=payload["doc_lengths"],
                avg_doc_length=payload["avg_doc_length"],
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise IndexingError(
                f"Invalid index format: {input_path}"
            ) from exc

        return chunks, lexical_index
