import sys
from src.cli import CLI

from src.domain.exceptions import RAGError

try:
    import fire
    import pydantic
    import tqdm
    import bm25s
    import torch
except ImportError as exc:
    missing_lib = exc.name
    print(f"❌ [ERROR] Missing dependency: '{missing_lib}'")
    print("Please install all required dependencies before running the system")
    print("Run 'pip install -e .' or 'make'")
    sys.exit(1)

# try:
#     from data.raw import
# except ImportError:
#     print("❌ [ERROR] Missing dependency: ")
#     sys.exit(1)


def main() -> None:
    try:
        fire.Fire(CLI)
    except ValueError as exc:
        print(f"❌ [VALUE ERROR] {exc}")
    except RAGError as exc:
        print(f"❌ [RAG ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ [UNEXPECTED ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()



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
