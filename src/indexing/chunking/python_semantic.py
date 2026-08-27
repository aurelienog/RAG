import ast
from dataclasses import dataclass

from ..python_ast import NodeSpan, PythonASTParser


@dataclass(frozen=True)
class SemanticUnit:
    """
    A semantically meaningful region of Python source.
    """

    node: ast.AST
    span: NodeSpan
    kind: str
    structural: bool


class PythonSemanticAnalyzer:
    """
    Converts a Python AST into semantic units suitable for chunking.
    """

    def __init__(self, parser: PythonASTParser, max_unit_size: int = 2000) -> None:
        self.parser = parser
        self.max_unit_size = max_unit_size

    def analyze(
        self,
        tree: ast.Module,
        content: str,
        line_offsets: list[int],
    ) -> list[SemanticUnit]:

        return self._analyze_node(
            node=tree,
            content=content,
            line_offsets=line_offsets,
        )

    def _analyze_node(
        self,
        node: ast.AST,
        content: str,
        line_offsets: list[int],
    ) -> list[SemanticUnit]:

        span = self.parser.get_span(
            node=node,
            line_offsets=line_offsets,
            content_length=len(content),
        )

        text = content[span.start:span.end]

        if not text.strip():
            return []

        if self._is_structural_node(node):
            return self._analyze_structural_node(
                node=node,
                span=span,
                text=text,
                content=content,
                line_offsets=line_offsets,
            )

        children = self._children(node)

        if not children:
            return [
                self._create_unit(
                    node=node,
                    span=span,
                    structural=False,
                )
            ]

        return self._analyze_children(
            children=children,
            content=content,
            line_offsets=line_offsets,
        )

    def _analyze_structural_node(
        self,
        node: ast.AST,
        span: NodeSpan,
        text: str,
        content: str,
        line_offsets: list[int],
    ) -> list[SemanticUnit]:

        # A small class/function remains one semantic unit.
        if len(text) <= self.max_unit_size:
            return [
                self._create_unit(
                    node=node,
                    span=span,
                    structural=True,
                )
            ]

        # Large classes/functions are recursively decomposed.
        children = self._children(node)

        if not children:
            return [
                self._create_unit(
                    node=node,
                    span=span,
                    structural=True,
                )
            ]

        return self._analyze_children(
            children=children,
            content=content,
            line_offsets=line_offsets,
        )

    def _analyze_children(
        self,
        children: list[ast.AST],
        content: str,
        line_offsets: list[int],
    ) -> list[SemanticUnit]:

        units: list[SemanticUnit] = []

        for child in children:
            units.extend(
                self._analyze_node(
                    node=child,
                    content=content,
                    line_offsets=line_offsets,
                )
            )

        return units

    def _children(self, node: ast.AST) -> list[ast.AST]:

        if isinstance(node, ast.Module):
            return list(node.body)

        if isinstance(node, ast.ClassDef):
            return list(node.body)

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return list(node.body)

        return []

    def _is_structural_node(self, node: ast.AST) -> bool:

        return isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )

    def _create_unit(
        self,
        node: ast.AST,
        span: NodeSpan,
        structural: bool,
    ) -> SemanticUnit:

        return SemanticUnit(
            node=node,
            span=span,
            kind=self._kind_for_node(node),
            structural=structural,
        )

    def _kind_for_node(self, node: ast.AST) -> str:

        if isinstance(node, ast.ClassDef):
            return "python_class"

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return "python_function"

        if isinstance(node, ast.Module):
            return "python_module"

        return "python_statement"