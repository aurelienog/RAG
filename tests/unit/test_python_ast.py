import ast

import pytest

from src.indexing.python_ast import NodeSpan, PythonASTParser


@pytest.fixture
def parser() -> PythonASTParser:
    return PythonASTParser()


def test_build_line_offsets_includes_end_of_source(
    parser: PythonASTParser,
) -> None:
    content = "one\ntwo\nthree"

    assert parser.build_line_offsets(content) == [0, 4, 8, 13]


def test_get_span_returns_exact_function_range(
    parser: PythonASTParser,
) -> None:
    content = "before = 1\ndef answer():\n    return 42\n"
    tree = parser.parse(content)
    line_offsets = parser.build_line_offsets(content)
    function = tree.body[1]

    span = parser.get_span(function, line_offsets, len(content))

    assert content[span.start:span.end] == "def answer():\n    return 42"


def test_module_span_covers_complete_source(
    parser: PythonASTParser,
) -> None:
    content = "value = 42\n"
    tree = parser.parse(content)

    span = parser.get_span(tree, parser.build_line_offsets(content), len(content))

    assert span == NodeSpan(start=0, end=len(content))


def test_parse_invalid_python_raises_syntax_error(
    parser: PythonASTParser,
) -> None:
    with pytest.raises(SyntaxError):
        parser.parse("def broken(:\n    pass\n")


def test_get_span_uses_child_nodes_when_position_is_missing(
    parser: PythonASTParser,
) -> None:
    content = "value = 42"
    node = ast.Module(body=[ast.parse(content).body[0]], type_ignores=[])
    node.body[0].lineno = None
    node.body[0].end_lineno = None

    span = parser.get_span(node.body[0], [0, len(content)], len(content))

    assert span == NodeSpan(start=0, end=len(content))
