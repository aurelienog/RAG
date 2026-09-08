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
    """Split Markdown content using structural separators and size limits.

    The chunker progressively applies Markdown structural separators to
    preserve meaningful sections. When structural splitting cannot produce
    chunks within the configured size limit, it falls back to line-based
    splitting.
    """

    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        """Split Markdown content into indexable chunks.

        Generated chunks preserve their original text and absolute character
        offsets within the source file. Empty or whitespace-only content
        produces no chunks.

        Args:
            file_path: Path of the Markdown file being chunked.
            content: Markdown source content to split.

        Returns:
            A list of Markdown chunks containing their source text and
            character offsets.
        """
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
        """Recursively split a content region using Markdown separators.

        The method applies separators in order of decreasing structural
        priority. If a region cannot be split using the remaining separators,
        line-based fallback splitting is used.

        Args:
            content: Complete Markdown source content.
            file_path: Path of the Markdown file being processed.
            start: Inclusive starting character offset of the region.
            end: Exclusive ending character offset of the region.
            separator_index: Index of the separator currently being applied.

        Returns:
            A list of ``(start, end)`` offset pairs representing the resulting
            chunks.
        """
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
        """Find separator boundaries within a content region.

        Args:
            content: Complete Markdown source content.
            start: Inclusive starting character offset of the region.
            end: Exclusive ending character offset of the region.
            separator: Separator string used to identify boundaries.

        Returns:
            A list of character offsets where the separator occurs within
            the specified region.
        """
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
        """Find all positions where a separator occurs in the content.

        Args:
            content: Text in which to search for the separator.
            start: Character offset from which the search begins.
            separator: String whose occurrences should be located.

        Returns:
            A list of character offsets corresponding to each occurrence of
            the separator.
        """
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
        """Merge consecutive spans while respecting the size limit.

        Args:
            spans: Offset pairs representing candidate content spans.

        Returns:
            A list of merged spans where no span exceeds
            ``max_chunk_size``.
        """
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
        """Split a content region using line-based fallback logic.

        Args:
            content: Complete Markdown source content.
            file_path: Path of the Markdown file being processed.
            start: Inclusive starting character offset of the region.
            end: Exclusive ending character offset of the region.

        Returns:
            A list of ``(start, end)`` offset pairs produced by line-based
            splitting.
        """
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
