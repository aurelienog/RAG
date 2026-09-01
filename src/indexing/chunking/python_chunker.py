from ...domain import Chunk
from ...config import DEFAULT_MAX_CHUNK_SIZE
from .python_ast import PythonASTParser
from .base import BaseChunker
from .fallback import split_lines


class PythonChunker(BaseChunker):
    """
    Chunk Python source around top-level AST nodes, with a cheap size fallback.
    """

    def __init__(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        parser: PythonASTParser | None = None,
    ) -> None:
        super().__init__(max_chunk_size)
        self.parser = parser or PythonASTParser()

    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        if not content.strip():
            return []

        if len(content) <= self.max_chunk_size:
            return [
                Chunk(
                    id=f"{file_path}_0_{len(content)}",
                    file_path=file_path,
                    text=content,
                    start=0,
                    end=len(content),
                    kind="python_module",
                )
            ]

        try:
            tree = self.parser.parse(content)
        except (SyntaxError, ValueError):
            return split_lines(
                text=content,
                file_path=file_path,
                start_offset=0,
                max_chunk_size=self.max_chunk_size,
                kind="python_syntax_fallback",
            )

        line_offsets = self.parser.build_line_offsets(content)
        chunks: list[Chunk] = []
        cursor = 0

        for node in tree.body:
            span = self.parser.get_span(
                node=node,
                line_offsets=line_offsets,
                content_length=len(content),
            )

            if span.start > cursor:
                chunks.extend(
                    self._split_region(
                        content[cursor:span.start],
                        file_path,
                        cursor,
                        "python_context",
                    )
                )

            node_text = content[span.start:span.end]
            if len(node_text) <= self.max_chunk_size:
                chunks.append(
                    Chunk(
                        id=f"{file_path}_{span.start}_{span.end}",
                        file_path=file_path,
                        text=node_text,
                        start=span.start,
                        end=span.end,
                        kind=self._kind_for_node(node),
                    )
                )
            else:
                chunks.extend(
                    self._split_region(
                        node_text,
                        file_path,
                        span.start,
                        "python_large_node",
                    )
                )

            cursor = span.end

        if cursor < len(content):
            chunks.extend(
                self._split_region(
                    content[cursor:],
                    file_path,
                    cursor,
                    "python_context",
                )
            )

        return chunks

    def _split_region(
        self,
        text: str,
        file_path: str,
        start_offset: int,
        kind: str,
    ) -> list[Chunk]:
        chunks = split_lines(
            text=text,
            file_path=file_path,
            start_offset=start_offset,
            max_chunk_size=self.max_chunk_size,
            kind=kind,
        )
        return [chunk for chunk in chunks if chunk.text.strip()]

    @staticmethod
    def _kind_for_node(node: object) -> str:
        node_name = type(node).__name__.lower()
        if node_name == "classdef":
            return "python_class"
        if node_name in {"functiondef", "asyncfunctiondef"}:
            return "python_function"
        return "python_statement"
