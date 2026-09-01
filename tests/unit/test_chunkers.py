import pytest

from src.indexing.chunking.base import BaseChunker
from src.indexing.chunking.fallback import split_lines
from src.indexing.chunking.markdown_chunker import MarkdownChunker
from src.indexing.chunking.python_chunker import PythonChunker


@pytest.mark.parametrize("chunker", [PythonChunker(20), MarkdownChunker(20)])
def test_empty_content_returns_no_chunks(chunker: BaseChunker) -> None:
    assert chunker.chunk_file("empty", "") == []


def test_split_lines_preserves_offsets_and_hard_splits_long_lines() -> None:
    content = "short\n" + "x" * 25 + "\nend"

    chunks = split_lines(
        text=content,
        file_path="data/raw/file.txt",
        start_offset=0,
        max_chunk_size=10,
        kind="test",
    )

    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert all(content[chunk.start:chunk.end] == chunk.text for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == content


def test_python_chunker_keeps_small_definitions_together() -> None:
    content = "import os\n\ndef answer():\n    return 42\n"

    chunks = PythonChunker(max_chunk_size=30).chunk_file("file.py", content)

    assert [chunk.kind for chunk in chunks] == [
        "python_statement",
        "python_function",
    ]
    # Quitamos el \n del final de la cadena esperada
    assert chunks[1].text == "def answer():\n    return 42"


# def test_python_chunker_keeps_small_definitions_together() -> None:
#     content = "import os\n\ndef answer():\n    return 42\n"

#     chunks = PythonChunker(max_chunk_size=200).chunk_file("file.py", content)

#     assert [chunk.kind for chunk in chunks] == [
#         "python_statement",
#         "python_function",
#     ]
#     assert chunks[1].text == "def answer():\n    return 42"


def test_python_chunker_falls_back_for_invalid_syntax() -> None:
    content = "def broken(:\n" + "x = 1\n" * 10

    chunks = PythonChunker(max_chunk_size=20).chunk_file("file.py", content)

    assert chunks
    assert all(chunk.kind == "python_syntax_fallback" for chunk in chunks)
    assert all(len(chunk.text) <= 20 for chunk in chunks)


def test_markdown_chunker_prefers_heading_boundaries() -> None:
    content = "# First\n\nfirst text\n\n# Second\n\nsecond text"

    chunks = MarkdownChunker(max_chunk_size=30).chunk_file("README.md", content)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("# First")
    assert chunks[1].text.lstrip().startswith("# Second")


def test_markdown_chunker_preserves_source_and_size_limit() -> None:
    content = "# Title\n\n" + "A paragraph with useful context. " * 20

    chunks = MarkdownChunker(max_chunk_size=40).chunk_file("README.md", content)

    assert chunks
    assert all(len(chunk.text) <= 40 for chunk in chunks)
    assert all(content[chunk.start:chunk.end] == chunk.text for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == content
