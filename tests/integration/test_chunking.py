from src.indexing.chunking.base import BaseChunker
from src.indexing.chunking.markdown_chunker import MarkdownChunker
from src.indexing.chunking.python_chunker import PythonChunker


def assert_chunk_contract(
    chunker: BaseChunker,
    file_path: str,
    content: str,
) -> None:
    chunks = chunker.chunk_file(file_path, content)

    assert chunks
    assert chunks == sorted(chunks, key=lambda chunk: chunk.start)
    assert all(0 <= chunk.start < chunk.end <= len(content) for chunk in chunks)
    assert all(len(chunk.text) <= chunker.max_chunk_size for chunk in chunks)
    assert all(content[chunk.start:chunk.end] == chunk.text for chunk in chunks)
    assert all(chunk.file_path == file_path for chunk in chunks)


def test_python_chunking_contract_on_mixed_source() -> None:
    content = '''"""Module documentation."""

class Search:
    def run(self, query: str) -> list[str]:
        return [query]


def helper() -> None:
    print("ready")
'''

    assert_chunk_contract(PythonChunker(50), "data/raw/search.py", content)


def test_markdown_chunking_contract_on_document() -> None:
    content = '''# Search

How retrieval works.

## Configuration

Set the index path and maximum chunk size.

```python
print("ready")
```
'''

    assert_chunk_contract(MarkdownChunker(45), "data/raw/README.md", content)
