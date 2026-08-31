from tqdm import tqdm

import json
from pathlib import Path

from ..config import DATA_PROCESSED, DATA_RAW, DEFAULT_MAX_CHUNK_SIZE
from ..domain import Chunk, IndexingError
from .chunking import PythonChunker, MarkdownChunker


class Indexer:
    def __init__(
            self,
            raw_dir: str | Path = DATA_RAW,
            processed_dir: str | Path = DATA_PROCESSED,
    ) -> None:

        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def index(
            self,
            max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, int]:

        if max_chunk_size <= 0 or max_chunk_size > 2000:
            raise IndexingError("max_chunk_size must be between 1 and 2000 characters.")

        if not self.raw_dir.exists():
            raise IndexingError(f"Input directory not found: {self.raw_dir}")
        if not self.raw_dir.is_dir():
            raise IndexingError(f"Expected a directory: {self.raw_dir}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[Chunk] = []
        files_indexed = 0

        for file_path in tqdm(self._iter_source_files(), desc="Indexing files"):
            try:
                content = self._read_text(file_path)
            except OSError as exc:
                raise IndexingError(f"Could not read file: {file_path}") from exc

            relative_path = self._to_relative_path(file_path, self.raw_dir)
            chunker = self._select_chunker(relative_path, max_chunk_size=max_chunk_size)
            file_chunks = chunker.chunk_file(relative_path, content)

            if file_chunks:
                chunks.extend(file_chunks)
                files_indexed += 1

        payload = {
            "chunks": [
                {
                    "id": chunk.id,
                    "file_path": chunk.file_path,
                    "text": chunk.text,
                    "start": chunk.start,
                    "end": chunk.end,
                    "kind": chunk.kind,
                }
                for chunk in chunks
            ]
        }
