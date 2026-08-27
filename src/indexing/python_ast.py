import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpan:
    """
    Character span of an AST node in the original source.

    The interval is half-open: [start, end).
    """

    start: int
    end: int


class PythonASTParser:
    """
    Parse Python source and resolve AST nodes to original source offsets.
    """

    def parse(self, content: str) -> ast.Module:
        """
        Parse Python source into an AST.

        Args:
            content: Complete Python source.

        Returns:
            Parsed Python module.

        Raises:
            SyntaxError: If the source cannot be parsed.
            ValueError: If the AST cannot be constructed.
        """
        return ast.parse(content)

    def build_line_offsets(self, content: str) -> list[int]:
        """
        Build absolute character offsets for the beginning of each line.

        The returned list is zero-based:

            offsets[0] -> start of line 1
            offsets[1] -> start of line 2
            ...

        An additional final offset is included for the end of the source.
        """
        offsets = [0]
        current_offset = 0

        for line in content.splitlines(keepends=True):
            current_offset += len(line)
            offsets.append(current_offset)

        return offsets

    def get_span(
        self,
        node: ast.AST,
        line_offsets: list[int],
        content_length: int,
    ) -> NodeSpan:
        """
        Return the character span occupied by an AST node.
        """
        if isinstance(node, ast.Module):
            return NodeSpan(
                start=0,
                end=content_length,
            )

        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        end_lineno = getattr(node, "end_lineno", None)
        end_col_offset = getattr(node, "end_col_offset", None)

        if lineno is not None and end_lineno is not None:
            start = line_offsets[lineno - 1]

            if col_offset is not None:
                start += col_offset

            end = line_offsets[end_lineno - 1]

            if end_col_offset is not None:
                end += end_col_offset

            return NodeSpan(
                start=max(0, start),
                end=min(end, content_length),
            )

        return self._get_fallback_span(
            node=node,
            line_offsets=line_offsets,
            content_length=content_length,
        )

    def _get_fallback_span(
        self,
        node: ast.AST,
        line_offsets: list[int],
        content_length: int,
    ) -> NodeSpan:
        start = content_length
        end = 0

        for child in ast.iter_child_nodes(node):
            child_span = self.get_span(
                node=child,
                line_offsets=line_offsets,
                content_length=content_length,
            )

            start = min(start, child_span.start)
            end = max(end, child_span.end)

        if start <= end:
            return NodeSpan(
                start=start,
                end=min(end, content_length),
            )

        return NodeSpan(
            start=0,
            end=content_length,
        )
