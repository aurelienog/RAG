from ...domain.chunk import Chunk
from ..python_ast import PythonASTParser
from .base import BaseChunker
from .chunk_packer import ChunkPacker
from .fallback import split_lines
from .python_semantic import PythonSemanticAnalyzer


class PythonChunker(BaseChunker):
    """
    Chunk Python source using AST-based semantic boundaries.
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        parser: PythonASTParser | None = None,
        semantic_analyzer: PythonSemanticAnalyzer | None = None,
        packer: ChunkPacker | None = None,
    ) -> None:
        super().__init__(max_chunk_size)

        self.parser = parser or PythonASTParser()

        self.semantic_analyzer = (
            semantic_analyzer
            or PythonSemanticAnalyzer(
                parser=self.parser,
                max_unit_size=max_chunk_size,
            )
        )

        self.packer = (
            packer
            or ChunkPacker(
                max_chunk_size=max_chunk_size,
            )
        )

    def chunk_file(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:

        if not content.strip():
            return []

        line_offsets = (
            self.parser.build_line_offsets(content)
        )

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

        units = self.semantic_analyzer.analyze(
            tree=tree,
            content=content,
            line_offsets=line_offsets,
        )

        return self.packer.pack(
            units=units,
            content=content,
            file_path=file_path,
        )
