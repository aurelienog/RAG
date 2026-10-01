from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from ..config import DATA_PROCESSED
from ..domain import Chunk


class QueryCache:
    """Persist query results to disk and reuse them across repeated searches.

    This class handles the serialization, deserialization, and integrity validation
    of search result chunks by associating them with a deterministic file key.

    Attributes:
        cache_dir (Path): The directory path where json cache files are stored.
    """

    def __init__(self, cache_dir: str | Path = DATA_PROCESSED / ".query_cache") -> None:
        """Initialize the query cache and ensure the storage directory exists.

        Args:
            cache_dir (str | Path): The directory path to use for storing cache files.
                Defaults to DATA_PROCESSED / ".query_cache".
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, query: str, k: int) -> str:
        """Build a deterministic SHA-256 key for the query and result size.

        The query is stripped, lowercased, and its internal whitespace is normalized
        before generating the hash to guarantee consistency across similar text inputs.

        Args:
            query (str): The search text string.
            k (int): The number of requested source documents.

        Returns:
            str: A hexadecimal string representing the SHA-256 hash of the request.

        Raises:
            ValueError: If the value of `k` is less than or equal to 0.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        normalized_query = " ".join(query.strip().lower().split())
        payload = f"{normalized_query}::{k}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path_for(self, query: str, k: int) -> Path:
        """Generate the absolute or relative file path for a specific cached query.

        Args:
            query (str): The search text string.
            k (int): The number of requested source documents.

        Returns:
            Path: The file path pointing to the expected JSON cache file location.
        """
        return self.cache_dir / f"{self.key_for(query, k)}.json"

    def set(self, query: str, k: int, results: list[Chunk]) -> list[Chunk]:
        """Persist the search results for the query to disk.

        Serializes a list of Chunk domain models into a formatted JSON file using
        their built-in serialization schemas.

        Args:
            query (str): The search text query.
            k (int): The number of requested source documents.
            results (list[Chunk]): The list of extracted text chunks to save.

        Returns:
            list[Chunk]: The exact same list of search results provided in the arguments.
        """
        cache_path = self._path_for(query, k)
        payload = [chunk.model_dump(mode="json") for chunk in results]

        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

        return results

    def get(self, query: str, k: int) -> list[Chunk]:
        """Load cached results if they exist, otherwise return an empty list.

        Reads and validates the stored JSON payload file, reconstructing the list
        of Chunk models. It ignores invalid schemas or read corruption gracefully.

        Args:
            query (str): The search text query.
            k (int): The number of requested source documents.

        Returns:
            list[Chunk]: A list of reconstructed text Chunks, or an empty list if
                the cache is missing, empty, or corrupted.

        Raises:
            ValueError: If the query parameter is an empty or whitespace-only string.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        cache_path = self._path_for(query, k)
        if not cache_path.exists():
            return []

        try:
            with cache_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(payload, list):
            return []

        results: list[Chunk] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                results.append(Chunk(**item))
            except TypeError:
                continue

        return results

    def contains(self, query: str, k: int) -> bool:
        """Return whether a cached result for the query exists on the filesystem.

        Args:
            query (str): The search text query.
            k (int): The number of requested source documents.

        Returns:
            bool: True if the corresponding cache file exists, False otherwise.
        """
        return self._path_for(query, k).exists()

    def clear(self) -> None:
        """Delete all cached query result files in the cache directory.

        Iterates through the target directory and unlinks every JSON cache entry safely.

        Returns:
            None
        """
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink(missing_ok=True)

    def get_or_compute(
        self,
        query: str,
        k: int,
        factory: Callable[[], list[Chunk]],
    ) -> list[Chunk]:
        """Return cached results or compute and persist them if missing.

        Acts as a standard cache-aside proxy pipeline wrapper. If a valid file is
        found, its data is loaded; otherwise, the fallback callable is executed
        and its output is serialized to disk before returning.

        Args:
            query (str): The search text query.
            k (int): The number of requested source documents.
            factory (Callable[[], list[Chunk]]): A execution fallback callback function
                that runs the original search pipeline when the cache misses.

        Returns:
            list[Chunk]: The active list of text chunks fetched from disk or newly generated.
        """
        cache_path = self._path_for(query, k)
        if cache_path.exists():
            cached = self.get(query, k)
            if cached is not None:
                return cached

        computed = factory()
        self.set(query, k, computed)
        return computed
