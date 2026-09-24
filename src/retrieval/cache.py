from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from ..config import DATA_PROCESSED
from ..domain import Chunk


class QueryCache:
    """Persist query results to disk and reuse them across repeated searches."""

    def __init__(self, cache_dir: str | Path = DATA_PROCESSED / ".query_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, query: str, k: int) -> str:
        """Build a deterministic SHA-256 key for the query and result size."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        normalized_query = " ".join(query.strip().lower().split())
        payload = f"{normalized_query}::{k}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path_for(self, query: str, k: int) -> Path:
        return self.cache_dir / f"{self.key_for(query, k)}.json"

    def set(self, query: str, k: int, results: list[Chunk]) -> list[Chunk]:
        """Persist the search results for the query to disk."""
        cache_path = self._path_for(query, k)
        payload = [chunk.model_dump(mode="json") for chunk in results]

        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

        return results

    def get(self, query: str, k: int) -> list[Chunk]:
        """Load cached results if they exist, otherwise return an empty list."""
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
        """Return whether a cached result for the query exists."""
        return self._path_for(query, k).exists()

    def clear(self) -> None:
        """Delete all cached query result files in the cache directory."""
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
        """Return cached results or compute and persist them."""
        cache_path = self._path_for(query, k)
        if cache_path.exists():
            cached = self.get(query, k)
            if cached is not None:
                return cached

        computed = factory()
        self.set(query, k, computed)
        return computed
