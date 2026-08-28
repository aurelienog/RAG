from ...domain.chunk import Chunk
from .base import BaseChunker
from .fallback import split_lines


MARKDOWN_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    " ",
    "",
]


class MarkdownChunker(BaseChunker):
    """Split Markdown using structural separators and a fixed size limit."""

    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        """Return Markdown chunks whose text and offsets match the source."""
        if not content.strip():
            return []

        spans = self._split_region(
            content,
            file_path,
            0,
            len(content),
            0,
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
        if end - start <= self.max_chunk_size:
            return [(start, end)]

        if separator_index >= len(MARKDOWN_SEPARATORS):
            return self._fallback_spans(
                content,
                file_path,
                start,
                end,
            )

        separator = MARKDOWN_SEPARATORS[separator_index]
        if not separator:
            return self._fallback_spans(
                content,
                file_path,
                start,
                end,
            )

        boundaries = [
            position
            for position in self._separator_positions(
                content,
                start,
                separator,
            )
            if start < position < end
        ]
        if not boundaries:
            return self._split_region(
                content,
                file_path,
                start,
                end,
                separator_index + 1,
            )

        spans: list[tuple[int, int]] = []
        section_start = start
        for boundary in boundaries + [end]:
            if boundary == section_start:
                continue
            spans.extend(
                self._split_region(
                    content,
                    file_path,
                    section_start,
                    boundary,
                    separator_index + 1,
                )
                if boundary - section_start > self.max_chunk_size
                else [(section_start, boundary)]
            )
            section_start = boundary

        return self._merge_spans(spans)

    @staticmethod
    def _separator_positions(
        content: str,
        start: int,
        separator: str,
    ) -> list[int]:
        positions: list[int] = []
        position = content.find(separator, start)
        while position != -1:
            positions.append(position)
            position = content.find(separator, position + len(separator))
        return positions

    def _merge_spans(
        self,
        spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if not merged or end - merged[-1][0] > self.max_chunk_size:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], end)
        return merged

    def _fallback_spans(
        self,
        content: str,
        file_path: str,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        chunks = split_lines(
            text=content[start:end],
            file_path=file_path,
            start_offset=start,
            max_chunk_size=self.max_chunk_size,
            kind="markdown_fallback",
        )
        return [(chunk.start, chunk.end) for chunk in chunks]