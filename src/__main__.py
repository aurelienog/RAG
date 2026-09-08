import sys
from src.cli import CLI

from src.domain.exceptions import RAGError

try:
    import fire
    import pydantic
    import tqdm
    import bm25s
    import torch
    import transformers
except ImportError as exc:
    missing_lib = exc.name
    print(f"❌ [ERROR] Missing dependency: '{missing_lib}'")
    print("Please install all required dependencies before running the system")
    print("Run 'pip install -e .' or 'make'")
    sys.exit(1)


def main() -> None:
    """Run the command-line interface and handle application errors.
    The CLI is executed through Google Fire. Expected value and application
    errors are reported to the user and terminate the process with a non-zero
    exit status. Unexpected exceptions are also caught to prevent raw
    tracebacks from being exposed to CLI users.

    Raises:
        SystemExit: If a dependency is missing or an application error occurs.
    """
    try:
        fire.Fire(CLI)
    except ValueError as exc:
        print(f"❌ [VALUE ERROR] {exc}")
        sys.exit(1)
    except RAGError as exc:
        print(f"❌ [RAG ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ [UNEXPECTED ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# 8. Hay un problema real con AnswerGenerator
# Esto sí lo revisaría.

# Tu código:

# inputs = self.tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     enable_thinking=False,
# )

# y luego:

# outputs = self.model.generate(
#     **inputs,

# Dependiendo de la versión de Transformers/model implementation, apply_chat_template() puede devolver un tensor directamente o un BatchEncoding.

# La documentación actual de Qwen muestra el patrón de usar return_dict=True y mover los inputs al dispositivo del modelo. 
# H
# Hugging Face

# Yo lo revisaría antes de la defensa.

# Además, si estás en CPU, algo como:

# inputs = inputs.to(self.model.device)

# puede ser importante si el modelo no está en el mismo dispositivo que los inputs.


# Tu BM25 lo implementaste tú:

# calculate_idf()
# score_bm25_term()

# y Retriever usa esas funciones.

# Por tanto:

# import bm25s

# parece innecesario.

# Y más importante: el subject dice:

# Beyond the tools required above, you may use any library you like.

# No necesitas bm25s para cumplir BM25.

# Yo eliminaría ese dependency check si bm25s tampoco está en tu pyproject.toml.



# Mi checklist del subject para tu proyecto
# De todo lo que has enseñado hasta ahora, yo revisaría especialmente estos puntos antes de seguir puliendo docstrings:

# ✅ Python 3.10+
# ⚠️ flake8
# ⚠️ mypy con exactamente los flags exigidos
# ✅ Pydantic para modelos
# ✅ Python Fire
# ✅ tqdm
# ✅ Qwen/Qwen3-0.6B
# ✅ Python chunking
# ✅ Markdown/text chunking
# ✅ BM25 implementado
# ✅ data/processed/
# ⚠️ exactitud de file_path
# ⚠️ límite absoluto de 2000 caracteres por chunk
# ✅ single-query search
# ✅ dataset search
# ✅ answer
# ✅ answer_dataset
# ✅ evaluate
# ⚠️ manejo de k=0
# ⚠️ empty/nonsense queries
# ⚠️ malformed JSON
# ⚠️ missing files
# ⚠️ generación sin traceback
# ⚠️ throughput ≤ 90 s / 200 preguntas
# ⚠️ indexing ≤ 5 min
# ⚠️ recall@5 ≥ 80% docs
# ⚠️ recall@5 ≥ 50% code
# ⚠️ Makefile con install, run, debug, clean, lint
# ⚠️ uv.lock
# ⚠️ README completo en inglés
# ⚠️ no incluir data/raw/vllm-0.10.1 en Git si es un archivo grande/proporcionado por el subject
# ⚠️ no importar/callar moulinette








#     try:
#         return pipeline(query)

#     except LLMERROR:
#         return format_retrieved_chunks(query)

#     except EmbeddingError:
#         return text_search(query)
#     except Exception:
#         return "Service temporarily unavailable. Please try again later"




# FALLBACK
# AST failure       → line-based chunking
# file failure      → skip + warning
# empty query       → empty results
# k <= 0            → empty results
# bad JSON          → controlled error
# no retrieval      → empty sources
# generation error  → deterministic fallback answer


# """

# #BONUS redis for caching?
# import numpy as np
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('qwen3..')

# sentences = ["blabla", "blablabla", "bla"]

# embeddings = model.encode(sentences)

# """
