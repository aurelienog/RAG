from __future__ import annotations

from pathlib import Path
from threading import Lock

from ..domain import Chunk
from .lexical_index import LexicalIndex
from .storage import IndexStorage


class IndexCache:
    """Keep already loaded indices in memory for the current process."""

    _cache: dict[Path, tuple[list[Chunk], LexicalIndex]] = {}
    _lock = Lock()

    @classmethod
    def _resolve_dir(
        cls,
        storage: IndexStorage | str | Path,
    ) -> Path:
        if isinstance(storage, IndexStorage):
            return storage.processed_dir

        return Path(storage)

    @classmethod
    def get(
        cls,
        storage: IndexStorage | str | Path,
    ) -> tuple[list[Chunk], LexicalIndex]:
        """Return a loaded index from memory, loading it once per processed dir."""
        processed_dir = cls._resolve_dir(storage)

        with cls._lock:
            cached = cls._cache.get(processed_dir)
            if cached is not None:
                return cached

            if isinstance(storage, IndexStorage):
                loaded = storage.load()
            else:
                loaded = IndexStorage(processed_dir).load()

            cls._cache[processed_dir] = loaded
            return loaded

    @classmethod
    def load(
        cls,
        storage: IndexStorage | str | Path,
    ) -> tuple[list[Chunk], LexicalIndex]:
        """Compatibility alias for callers that expect a ``load`` method."""
        return cls.get(storage)

    @classmethod
    def clear(cls) -> None:
        """Clear the in-memory index cache."""
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def contains(cls, storage: IndexStorage | str | Path) -> bool:
        """Return whether the processed directory has a cached index."""
        processed_dir = cls._resolve_dir(storage)
        with cls._lock:
            return processed_dir in cls._cache
