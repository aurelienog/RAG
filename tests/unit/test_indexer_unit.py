import ast
import pytest
from pathlib import Path

# Ajusta los imports según la estructura real de tus paquetes locales
from src.utils import Tokenizer
from src.indexing.chunking import PythonASTParser
from src.indexing.indexer import Indexer


# === TESTS PARA EL TOKENIZER ===
def test_tokenizer_lowercases_and_removes_stopwords():
    text = "The AsyncFunctionDef is inside a Class"
    tokens = Tokenizer.tokenize(text)

    # Tu Regex separa CamelCase de manera excelente:
    assert "async" in tokens
    assert "function" in tokens
    assert "def" in tokens
    assert "inside" in tokens
    assert "class" in tokens

    # Las stopwords deben seguir quedando fuera
    assert "the" not in tokens
    assert "a" not in tokens


def test_tokenizer_camel_case_splitting():
    text = "vllmModelExecutor"
    tokens = Tokenizer.tokenize(text)
    assert "vllm" in tokens
    assert "model" in tokens
    assert "executor" in tokens


# === TESTS PARA EL PARSER AST PYTHON ===
def test_python_ast_parser_builds_correct_offsets():
    parser = PythonASTParser()
    content = "line1\nline2\nline3"
    # line1\n = 6 chars, line2\n = 6 chars, line3 = 5 chars
    offsets = parser.build_line_offsets(content)
    assert offsets == [0, 6, 12, 17]


def test_python_ast_parser_get_span_with_decorator():
    parser = PythonASTParser()
    content = (
        "@classmethod\n"
        "def my_method(cls):\n"
        "    pass\n"
    )
    tree = ast.parse(content)
    # El primer nodo del body es la función
    func_node = tree.body[0]
    line_offsets = parser.build_line_offsets(content)

    span = parser.get_span(func_node, line_offsets, len(content))

    # Tu código calcula con precisión quirúrgica:
    # El caracter 0 es '@', el nodo del decorador reporta col_offset=1 (empieza en 'c')
    assert span.start == 1


# === TESTS UNITARIOS PARA EL INDEXER ===
def test_indexer_invalid_max_chunk_size(tmp_path):
    indexer = Indexer(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "proc")
    with pytest.raises(Exception) as exc_info:
        indexer.index(max_chunk_size=3000)  # Límite superior de 2000
    assert "max_chunk_size must be between 1 and 2000" in str(exc_info.value)


def test_indexer_to_project_relative_path(monkeypatch, tmp_path):
    # Forzamos a Path.cwd() a apuntar a nuestra raíz de test simulada
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    file_path = tmp_path / "data" / "raw" / "vllm-corp" / "README.md"
    rel_path = Indexer._to_project_relative_path(file_path)

    assert rel_path == "data/raw/vllm-corp/README.md"
