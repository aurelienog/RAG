from ...domain import Chunk
from ...config import DEFAULT_MAX_CHUNK_SIZE
from .python_ast import PythonASTParser
from .base import BaseChunker
from .fallback import split_lines


class PythonChunker(BaseChunker):
    """Split Python source around top-level AST nodes.

    The chunker preserves Python structural boundaries when possible and
    falls back to line-based splitting for large nodes or invalid Python
    syntax.
    """

    def __init__(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        parser: PythonASTParser | None = None,
    ) -> None:
        """Initialize the Python chunker.

        Args:
            max_chunk_size: Maximum number of characters allowed in each
                generated chunk.
            parser: Optional parser used to parse Python source and resolve
                AST node positions. If omitted, a new ``PythonASTParser`` is
                created.

        Raises:
            IndexingError: If ``max_chunk_size`` is outside the allowed
                range defined by ``BaseChunker``.
        """

        super().__init__(max_chunk_size)
        self.parser = parser or PythonASTParser()

    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        """Split Python source into indexable chunks.

        Top-level AST nodes are kept together when they fit within the
        configured size limit. Large nodes and source regions between nodes
        are split using line-based fallback logic. If the source cannot be
        parsed, the entire file is processed using the same line-based
        fallback.

        Args:
            file_path: Path of the Python file being chunked.
            content: Python source content to split.

        Returns:
            A list of chunks containing the source text, absolute character
            offsets, and a type describing the kind of Python content.

        """

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
        """Split a source region using line-based fallback logic.

        Empty or whitespace-only chunks are removed from the result.

        Args:
            text: Source region to split.
            file_path: Path of the source file containing the region.
            start_offset: Absolute character offset where the region begins.
            kind: Type or category assigned to each generated chunk.

        Returns:
            A list of non-empty chunks produced from the source region.
        """

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
        """Determine the chunk type corresponding to an AST node.

        Args:
            node: AST node whose type should be classified.

        Returns:
            ``"python_class"`` for class definitions,
            ``"python_function"`` for synchronous or asynchronous function
            definitions, or ``"python_statement"`` for other node types.
        """

        node_name = type(node).__name__.lower()
        if node_name == "classdef":
            return "python_class"
        if node_name in {"functiondef", "asyncfunctiondef"}:
            return "python_function"
        return "python_statement"
