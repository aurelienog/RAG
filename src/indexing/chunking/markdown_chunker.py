from ...domain import Chunk
from .base import BaseChunker
from .fallback import split_lines


MARKDOWN_SEPARATORS = (
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
)


class MarkdownChunker(BaseChunker):
    """Split Markdown using structural separators and a fixed size limit."""

    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        """Return Markdown chunks preserving source text and offsets."""
        if not content.strip():
            return []

        spans = self._split_region(
            content=content,
            file_path=file_path,
            start=0,
            end=len(content),
            separator_index=0,
        )

        return [
            Chunk(
                id=f"{file_path}_{start}_{end}",
                file_path=file_path,
                text=content[start:end],
                start=start,
                end=end,
                kind="markdown",
            )
            for start, end in spans
            if content[start:end].strip()
        ]

    def _split_region(
        self,
        content: str,
        file_path: str,
        start: int,
        end: int,
        separator_index: int,
    ) -> list[tuple[int, int]]:
        """Recursively split a region using Markdown separators."""
        if end - start <= self.max_chunk_size:
            return [(start, end)]

        if separator_index >= len(MARKDOWN_SEPARATORS):
            return self._fallback_spans(
                content=content,
                file_path=file_path,
                start=start,
                end=end,
            )

        separator = MARKDOWN_SEPARATORS[separator_index]

        boundaries = self._find_boundaries(
            content=content,
            start=start,
            end=end,
            separator=separator,
        )

        if not boundaries:
            return self._split_region(
                content=content,
                file_path=file_path,
                start=start,
                end=end,
                separator_index=separator_index + 1,
            )

        spans: list[tuple[int, int]] = []
        section_start = start

        for boundary in boundaries + [end]:
            if boundary <= section_start:
                continue

            if boundary - section_start > self.max_chunk_size:
                spans.extend(
                    self._split_region(
                        content=content,
                        file_path=file_path,
                        start=section_start,
                        end=boundary,
                        separator_index=separator_index + 1,
                    )
                )
            else:
                spans.append(
                    (section_start, boundary)
                )

            section_start = boundary

        return self._merge_spans(spans)

    def _find_boundaries(
        self,
        content: str,
        start: int,
        end: int,
        separator: str,
    ) -> list[int]:
        """Find valid separator boundaries inside the current region."""
        return [
            position
            for position in self._separator_positions(
                content=content,
                start=start,
                separator=separator,
            )
            if start < position < end
        ]

    @staticmethod
    def _separator_positions(
        content: str,
        start: int,
        separator: str,
    ) -> list[int]:
        """Return all positions where the separator occurs."""
        positions: list[int] = []

        position = content.find(separator, start)

        while position != -1:
            positions.append(position)

            position = content.find(
                separator,
                position + len(separator),
            )

        return positions

    def _merge_spans(
        self,
        spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Merge consecutive spans without exceeding max_chunk_size."""
        merged: list[tuple[int, int]] = []

        for start, end in spans:
            if not merged:
                merged.append((start, end))
                continue

            merged_start = merged[-1][0]

            if end - merged_start <= self.max_chunk_size:
                merged[-1] = (merged_start, end)
            else:
                merged.append((start, end))

        return merged

    def _fallback_spans(
        self,
        content: str,
        file_path: str,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        """Fall back to line-based splitting."""
        chunks = split_lines(
            text=content[start:end],
            file_path=file_path,
            start_offset=start,
            max_chunk_size=self.max_chunk_size,
            kind="markdown_fallback",
        )

        return [
            (chunk.start, chunk.end)
            for chunk in chunks
        ]
