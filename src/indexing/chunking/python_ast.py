import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpan:
    """Represent the character span of an AST node in the original source.

    The span uses a half-open interval ``[start, end)``.

    Attributes:
        start: Inclusive starting character offset of the AST node.
        end: Exclusive ending character offset of the AST node.
    """

    start: int
    end: int


class PythonASTParser:
    """Parse Python source and resolve AST nodes to source offsets.

    This parser uses Python's abstract syntax tree to identify source regions
    corresponding to AST nodes and converts line and column positions into
    absolute character offsets.
    """

    def parse(self, content: str) -> ast.Module:
        """Parse Python source code into an abstract syntax tree.

        Args:
            content: Complete Python source code to parse.

        Returns:
            The parsed Python module represented as an AST.

        Raises:
            SyntaxError: If the source code contains invalid Python syntax.
            ValueError: If the AST cannot be constructed.
        """
        return ast.parse(content)

    def build_line_offsets(self, content: str) -> list[int]:
        """Build absolute offsets for the beginning of each source line.

        The returned list is zero-based, where ``offsets[0]`` is the start
        of the first line and each subsequent element represents the start
        of the corresponding line. A final offset is included to represent
        the end of the source.

        Args:
            content: Complete source content.

        Returns:
            A list of absolute character offsets for the beginning of each
            line, including the final offset at the end of the source.
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
        """Return the character span occupied by an AST node.

        Function, asynchronous function, and class nodes include their first
        decorator in the returned span when decorators are present. Nodes
        without explicit source position information are resolved using
        their child nodes.

        Args:
            node: AST node whose source span should be resolved.
            line_offsets: Absolute character offsets for the beginning of
                each source line.
            content_length: Total number of characters in the source.

        Returns:
            A ``NodeSpan`` representing the node's half-open character span
            in the original source.
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
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if hasattr(node, "decorator_list") and node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    lineno = getattr(first_decorator, "lineno", lineno)
                    col_offset = getattr(first_decorator, "col_offset", col_offset)

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
        """Resolve a node's span from the spans of its child nodes.

        This fallback is used when an AST node does not provide explicit
        line and column position information.

        Args:
            node: AST node whose child spans should be inspected.
            line_offsets: Absolute character offsets for the beginning of
                each source line.
            content_length: Total number of characters in the source.

        Returns:
            A ``NodeSpan`` covering all child nodes. If no valid child span
            can be determined, the complete source span is returned.
        """

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
