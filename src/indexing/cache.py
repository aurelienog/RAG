from __future__ import annotations

from pathlib import Path
from threading import Lock

from ..domain import Chunk
from .lexical_index import LexicalIndex
from .storage import IndexStorage


class IndexCache:
    """Keep already loaded indices in memory for the current process.

    This cache ensures that identical index storage paths are not loaded into memory
    multiple times concurrently across different threads.

    Attributes:
        _cache (dict[Path, tuple[list[Chunk], LexicalIndex]]): The internal dictionary
            mapping processed directory paths to their cached chunk list and lexical index.
        _lock (Lock): A thread lock guarding concurrent access and mutations
        to the internal cache map.
    """

    _cache: dict[Path, tuple[list[Chunk], LexicalIndex]] = {}
    _lock = Lock()

    @classmethod
    def _resolve_dir(
        cls,
        storage: IndexStorage | str | Path,
    ) -> Path:
        """Resolve any supported storage reference format into a definitive Path directory object.

        Args:
            storage (IndexStorage | str | Path): An instance of IndexStorage,
            or a direct string/Path representation pointing to the target storage location.

        Returns:
            Path: The resolved absolute or relative path to the processed directory.
        """
        if isinstance(storage, IndexStorage):
            return storage.processed_dir

        return Path(storage)

    @classmethod
    def get(
        cls,
        storage: IndexStorage | str | Path,
    ) -> tuple[list[Chunk], LexicalIndex]:
        """Return a loaded index from memory, loading it once per processed dir.

        If the index is already stored in the cache dictionary, it returns the stored instance
        immediately. If not, it uses thread locking to prevent race conditions, reads the data
        from disk using the IndexStorage system, stores the output, and returns it.

        Args:
            storage (IndexStorage | str | Path): The storage system wrapper or the path
                pointing to the index to look up or load.

        Returns:
            tuple[list[Chunk], LexicalIndex]: A tuple containing the list of processed text
                Chunks and its corresponding LexicalIndex.
        """
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
        """Compatibility alias for callers that expect a ``load`` method.

        Acts as a direct pass-through pipeline to the main `get` method logic.

        Args:
            storage (IndexStorage | str | Path): The storage reference or path string/Path
                representing the index object to retrieve.

        Returns:
            tuple[list[Chunk], LexicalIndex]: The extracted chunks and lexical index tuple mappings.
        """
        return cls.get(storage)

    @classmethod
    def clear(cls) -> None:
        """Clear the in-memory index cache.

        Uses the internal thread lock to securely wipe all reference entries from the internal cache
        dictionary in a thread-safe manner.

        Returns:
            None
        """
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def contains(cls, storage: IndexStorage | str | Path) -> bool:
        """Return whether the processed directory has a cached index in memory.

        Args:
            storage (IndexStorage | str | Path): The storage instance or filesystem path to check
                against the cache.

        Returns:
            bool: True if the index exists within the active memory cache pool, False otherwise.
        """
        processed_dir = cls._resolve_dir(storage)
        with cls._lock:
            return processed_dir in cls._cache
