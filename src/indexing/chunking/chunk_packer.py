from ...domain.chunk import Chunk
from .fallback import split_lines
from .python_semantic import SemanticUnit


class ChunkPacker:
    """
    Packs semantic units into chunks without exceeding max_chunk_size.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0"
            )

        self.max_chunk_size = max_chunk_size

    def pack(
        self,
        units: list[SemanticUnit],
        content: str,
        file_path: str,
    ) -> list[Chunk]:

        chunks: list[Chunk] = []
        pending: list[SemanticUnit] = []

        for unit in units:

            if unit.structural:
                chunks.extend(
                    self._flush_pending(
                        pending,
                        content,
                        file_path,
                    )
                )
                pending = []

                chunks.extend(
                    self._pack_structural(
                        unit,
                        content,
                        file_path,
                    )
                )
                continue

            if self._fits_with(
                unit,
                pending,
                content,
            ):
                pending.append(unit)
                continue

            chunks.extend(
                self._flush_pending(
                    pending,
                    content,
                    file_path,
                )
            )

            pending = [unit]

        chunks.extend(
            self._flush_pending(
                pending,
                content,
                file_path,
            )
        )

        return chunks

    def _pack_structural(
        self,
        unit: SemanticUnit,
        content: str,
        file_path: str,
    ) -> list[Chunk]:

        text = content[
            unit.span.start:unit.span.end
        ]

        if len(text) <= self.max_chunk_size:
            return [
                self._create_chunk(
                    unit,
                    content,
                    file_path,
                )
            ]

        return split_lines(
            text=text,
            file_path=file_path,
            start_offset=unit.span.start,
            max_chunk_size=self.max_chunk_size,
            kind=f"{unit.kind}_line_fallback",
        )

    def _fits_with(
        self,
        unit: SemanticUnit,
        pending: list[SemanticUnit],
        content: str,
    ) -> bool:

        if not pending:
            return True

        start = pending[0].span.start
        end = unit.span.end

        return (
            end - start
            <= self.max_chunk_size
        )

    def _flush_pending(
        self,
        pending: list[SemanticUnit],
        content: str,
        file_path: str,
    ) -> list[Chunk]:

        if not pending:
            return []

        start = pending[0].span.start
        end = pending[-1].span.end

        text = content[start:end]

        if not text.strip():
            return []

        return [
            Chunk(
                id=f"{file_path}_{start}_{end}",
                file_path=file_path,
                text=text,
                start=start,
                end=end,
                kind="python_statements",
            )
        ]

    def _create_chunk(
        self,
        unit: SemanticUnit,
        content: str,
        file_path: str,
    ) -> Chunk:

        return Chunk(
            id=(
                f"{file_path}_"
                f"{unit.span.start}_"
                f"{unit.span.end}"
            ),
            file_path=file_path,
            text=content[
                unit.span.start:unit.span.end
            ],
            start=unit.span.start,
            end=unit.span.end,
            kind=unit.kind,
        )
