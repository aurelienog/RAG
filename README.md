*This project has been created as part of the 42 curriculum by aunoguei.*

# RAG against the machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about a software codebase.

The system follows the classic RAG workflow:

1. Index a source tree into structured text/code chunks.
2. Retrieve the most relevant chunks for a user question.
3. Generate an answer with a local causal language model using only the retrieved context.
4. Evaluate retrieval quality with Recall@k against a ground-truth dataset.

Indexing is incremental (Bonus nº3), when a file changes, re-index only that file instead of rebuilding the whole index.

The implementation supports three retrieval modes:

- BM25 lexical retrieval for exact terms and keyword matching.

- Semantic retrieval using Sentence Transformers and cosine similarity. (Bonus nº1)

- Hybrid retrieval combining BM25 and semantic rankings with Reciprocal Rank Fusion (RRF). (Bonus nº2)

The default CLI commands use BM25 retrieval. Semantic and hybrid retrieval are available through dedicated commands.

The codebase is designed to index source-oriented repositories, with dedicated chunking strategies for Python and Markdown/text-like files. Indexes are persisted so that search and answer generation can be performed without rebuilding the lexical index every time.

## Instructions

### Requirements

The project requires Python and the libraries used by the implementation, including:

- Python Fire
- Pydantic
- tqdm
- NumPy
- PyTorch
- Transformers
- Sentence Transformers
- flake8
- mypy

### Installation

#### Input data

By default, the indexer reads:
```data/raw/vllm-0.10.1/```

and writes its processed index to:
```data/processed/```

The source directory is expected to contain supported source files. The indexer accepts:
```
.py .md .markdown .txt .rst .yaml .yml .toml
.c .cpp .h .hpp .cu
```
Several generated/dependency directories are ignored, including ```.git```, virtual environments, ```__pycache__```, ```build```, ```dist```, ```node_modules``` and related directories.

### Compilation

makefile or CLI

### Execution

## Resources

- [AWS — What is RAG ?](https://aws.amazon.com/fr/what-is/retrieval-augmented-generation/)  

- [freeCodeCamp — RAG & MCP fundamentals?](https://www.youtube.com/watch?v=I7_WXKhyGms)

- [Python Fire - Using a Fire CLI](https://google.github.io/python-fire/guide/)

- [Tqdm Python](https://www.datacamp.com/tutorial/tqdm-python)

- [uuid Module](https://www.w3schools.com/python/ref_module_uuid.asp)

- [AST](https://earthly.dev/blog/python-ast/)

- [Chunking strategies guide](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)

- [FastAPI](https://kinsta.com/es/blog/fastapi/)


### AI Usage

AI tools were used as a development aid for this project, primarily for:

discussing RAG architecture and alternative retrieval designs;

reviewing implementation choices such as BM25, semantic retrieval and RRF;

identifying edge cases around chunking, indexing, persisted embeddings and CLI behavior;

helping review documentation structure and README completeness;

assisting with explanations and wording during development.

AI-generated suggestions were reviewed and adapted to the actual implementation.


## System Architecture

The main pipeline is:

```
```

### Main components

- ```src/indexing/```: source discovery, chunking, lexical indexing, persistence and incremental indexing.

- ```src/retrieval/```: BM25, semantic retrieval, hybrid retrieval and search orchestration.

- ```src/generation/```: prompt construction, local LLM generation and answer orchestration.

- ```src/ingest.py```: JSON input/output contracts, source resolution and context construction.

- ```src/evaluation.py```: retrieval evaluation with Recall@k.

- ```src/cli.py```: command-line operations exposed through Python Fire.

- ```src/domain/``` and ```src/models.py```: domain objects and validated JSON models.


## Chunking Strategy

The default maximum chunk size is 2,000 characters (```DEFAULT_MAX_CHUNK_SIZE```).

The project uses different strategies depending on the source format.

### Python

Python files are parsed using the Python AST when possible.

- Files shorter than the maximum size remain a single chunk.

- Top-level AST nodes are kept together when they fit within the limit.

- Functions and classes are therefore preferentially kept as coherent units.

- Large AST nodes are split with a line-based fallback.

- Source regions between top-level nodes are also split line-by-line.

- If Python parsing fails because of invalid syntax, the complete file falls back to line-based chunking.

- Extremely long individual lines are hard-split so the 2,000-character limit is never exceeded.

Each chunk keeps its original file path and absolute character offsets.

### Markdown and text-like files

Markdown is split progressively using structural separators:

1. "#" headings
2. "##" headings
3. "###" headings
4. "####" headings
5. paragraph boundaries
6. line boundaries

When structural splitting is insufficient, the implementation falls back to line-based splitting and, for oversized individual lines, fixed-size hard splitting.

This strategy aims to preserve semantic/document structure before sacrificing it for the hard size limit.

### Chunk identity

Chunks use an identifier derived from:

```file_path + start_offset + end_offset```

The indexer also stores SHA-256 hashes of source files. If a file has not changed since the previous indexing run, its existing chunks can be reused instead of being regenerated.

## Retrieval Method

### BM25 retrieval

The lexical retriever builds an inverted index containing:
- term -> postings list
- term document frequency
- token count per chunk
- average chunk length

Queries are tokenized with the project's tokenizer. For every query term present in the index, the implementation calculates its BM25 contribution using:
- ```k1 = 1.5```
- ```b = 0.75```

The final BM25 score for a chunk is the sum of the contributions of its recognized query terms. Candidates are returned in descending BM25 score order.

This approach is useful when a question contains exact identifiers, API names, filenames or technical terminology.

### Semantic retrieval

The semantic retriever uses the Sentence Transformers model:

```all-MiniLM-L6-v2```

Chunk texts are encoded into normalized dense vectors. The query is encoded in the same space, and similarity is calculated with a dot product between normalized vectors, which is equivalent to cosine similarity.

Embeddings are persisted in:

```data/processed/embeddings.npy```  
```data/processed/embeddings_ids.json```

The explicit chunk-ID mapping prevents embedding rows from becoming detached from their source chunks.

### Hybrid retrieval and ranking

Hybrid retrieval obtains candidate lists from both BM25 and semantic search. By default it retrieves max(k * 4, 10) candidates from each retriever.

The two rankings are combined with ```Reciprocal Rank Fusion (RRF)```:

```
RRF score = 1 / (rrf_k + rank + 1)

with:
rrf_k = 60
```
If a chunk appears in both rankings, its contributions are added. The final candidates are sorted by their combined RRF score and the top k chunks are returned.

## Performance Analysis

The project evaluates retrieval using:
```
Recall@1
Recall@3
Recall@5
Recall@10
```
A retrieved source is considered to match a ground-truth source when:
- it belongs to the same file; and
- the intersection-over-union (IoU) of their character ranges is at least ```0.05```.

For each question, recall is calculated as the fraction of expected sources matched by the top ```k``` retrieved sources, then averaged across matching questions.

The evaluator therefore measures **source retrieval quality**, not the factual quality of generated answers.

### Retrieval Benchmark

The retrieval system was evaluated on the AnsweredQuestions datasets.


```
Test

Result

Indexing time

32 s

Warm retrieval: 200 questions (docs + code)

20 s

Docs Recall@1

59.0%

Docs Recall@3

75.0%

Docs Recall@5

81.0%

Docs Recall@10

83.0%

Code Recall@1

35.0%

Code Recall@3

53.0%

Code Recall@5

58.0%

Code Recall@10

68.0%
```

The examination contained 100 questions for the documentation dataset and 100 questions for the code dataset. All 200 questions had valid student sources.

The required Recall@5 thresholds were 80% for documentation and 50% for code. The system achieved 81.0% on documentation and 58.0% on code, so both retrieval requirements were met.

### Performance interpretation

The results show that the retriever is substantially better at identifying relevant documentation than code at low values of ```k```, while increasing ```k``` improves coverage for both datasets. For documentation, Recall@5 reaches 81.0% and Recall@10 reaches 83.0%, indicating that most relevant sources are already present among the first few retrieved results. Code retrieval has lower early-rank recall, but improves from 35.0% at Recall@1 to 68.0% at Recall@10.

### Benchmark limitations

The benchmark evaluates retrieval quality against a private dataset and therefore does not measure answer-generation quality directly. Recall@k indicates whether the expected source appears in the top-k retrieved results; it does not by itself measure the factual correctness, completeness, or clarity of the final generated answer.

### How to obtain the project's actual scores



### Runtime considerations

- BM25 uses an inverted index, avoiding a full text comparison against every chunk for each query.
- Semantic retrieval performs a dense matrix similarity search over all stored embeddings, so query-time cost grows with the number of indexed chunks.
- Semantic indexing is more expensive than lexical indexing because every chunk must be encoded.
- Embeddings are persisted and reused on subsequent semantic searches.
- Incremental indexing uses SHA-256 file hashes and reuses chunks from unchanged files.
- Answer generation adds the cost of loading and running the local language model.

### Generation

The answer generator uses:

```Qwen/Qwen3-0.6B```

through Hugging Face Transformers.

The prompt explicitly instructs the model to:

- use only the retrieved context;
- avoid inventing unsupported facts;
- state that the information is unavailable when the retrieved sources are insufficient.

Retrieved chunks are converted into a bounded context of at most **12,000 characters**. Each context section contains the source file path followed by the chunk text.

Generation is deterministic ```(do_sample=False)``` with a default maximum of 512 newly generated tokens.

If no relevant chunks are retrieved, the service returns a deterministic message instead of attempting unsupported generation.

## Design decisions: Explain key implementation choices



## Challenges faced: Document difficulties encountered and solutions

## Example usage

All input and output paths are configurable CLI arguments.

1. Build the lexical index
```
uv run python -m src index --max_chunk_size 2000
```

2. Search with BM25
```
uv run python -m src search "What models does vLLM support?"
```

3. Generate an answer
```
uv run python -m src answer "How to deploy vLLM?"
```

4. Search a dataset

The public datasets share file names, so writing every run into
the same folder would overwrite previous results.

```
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```

5. Generate answers for an existing result set
```
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

6. Evaluate retrieval
```
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
```

7. Evaluate 
```
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Alternatively (BONUS):

1. Build the semantic index
```
uv run python -m src index_semantic
```

2. Search semantically
```
uv run python -m src search_semantic
```

3. Search with hybrid retrieval
```
uv run python -m src search_hybrid
```


## Bonus Features

### 1. Semantic embeddings

### 2. Hybrid retrieval

### 3. Incremental indexing

#### File indexing

The indexer maintains a manifest of the files that were processed previously. Each file is identified by a SHA-256 hash of its contents. During a new indexing run, the current files are compared with the previous manifest and classified as:

Unchanged: the file content is identical, so its existing chunks can be reused.

Modified: the file content changed, so the file is chunked again.

New: the file did not exist in the previous manifest, so it is chunked and indexed.

Deleted: the file is no longer present and its previous chunks are removed from the index.

This makes re-indexing proportional to the changes in the corpus instead of requiring every document to be processed again.

```
                 Raw files
                     │
                   SHA-256
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      unchanged   modified     new
          │          │          │
          ▼          ▼          ▼
    reuse chunks  re-chunk    re-chunk
          │          │          │
          └──────────┴──────────┘
                     │
                     ▼
              Updated index
```

A representative indexing run produced the following statistics:
```
=== Indexing statistics ===  

Files indexed:     2202  
Unchanged files:   2201  
Modified files:       1  
New files:            0  
Deleted files:        0  
Chunks:           40915  
```
In this run, 2,201 of 2,202 files were unchanged, while only one file required modification processing. No new or deleted files were detected. The final index contained 40,915 chunks, illustrating the benefit of incremental processing on a large corpus when most files remain unchanged.

#### Incremental embedding generation

Incremental indexing also avoids recomputing embeddings for chunks that have not changed. After the final set of chunks has been determined, embeddings are split into two paths:

embeddings belonging to reused chunks are kept;

only new or modified chunks are encoded again.

This reduces the most expensive part of semantic indexing when the corpus changes only partially.
```
                    index()
                       │
                    manifest
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      unchanged     modified       new
          │            │            │
          │         re-chunk       chunk
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Final chunks
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          reuse                 encode
      old embeddings        changed chunks
             │                   │
             └─────────┬─────────┘
                       ▼
                  Final index
```

#### Efficient incremental manifest management

The manifest acts as the source of truth for incremental updates. Instead of comparing complete document contents or rebuilding the vector index blindly, the indexer stores file-level metadata and uses SHA-256 content hashes to detect changes.

For each indexing run, the manifest allows the system to determine the minimum required work:

```
File state

Chunking

Embedding

Index action

Unchanged

Reuse

Reuse

Keep existing entries

Modified

Re-run

Recompute changed chunks

Replace old entries

New

Run

Compute

Add entries

Deleted

None

None

Remove old entries

This design provides both correctness and efficiency: changes are detected deterministically, while unaffected data remains available for reuse.
```

### Caching

This project includes two complementary cache layers to reduce repeated startup and retrieval overhead.

#### 1. Index cache

The in-memory index cache keeps the loaded BM25 index alive inside the same Python process, so repeated loads of the same processed index do not deserialize the same JSON payload again.

This is especially useful when the application reuses the same dataset across multiple searches in the same runtime.

#### 2. Query cache

The query cache persists retrieval results on disk using a deterministic SHA-256 key derived from the normalized query text and the requested result count. This allows repeated queries to reuse previously computed results instead of re-running the search.

This improves cold-start behavior and reduces latency for repeated same-query lookups.

#### Demonstration

Run the following snippet in a single Python process to compare cold and cached access:

```
uv run python - <<'PY'
import time
from pathlib import Path

from src.indexing.cache import IndexCache
from src.indexing.storage import IndexStorage
from src.retrieval.cache import QueryCache
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.search_service import SearchService

processed_dir = Path("data/processed")
query_cache_dir = Path("data/query_cache")

# --- Index cache demo ---
storage = IndexStorage(processed_dir)
IndexCache.clear()

t0 = time.perf_counter()
idx1 = IndexCache.get(storage)
t1 = time.perf_counter()

t2 = time.perf_counter()
idx2 = IndexCache.get(storage)
t3 = time.perf_counter()

print(f"Cold load:   {(t1 - t0):.5f} s")
print(f"Cached load: {(t3 - t2):.7f} s")
print(f"Same object in RAM (idx1 is idx2): {idx1 is idx2}")

# --- Query cache demo ---
query_cache = QueryCache(query_cache_dir)
query_cache.clear()

query = "what is rag"
k = 5

print(f"SHA-256 key: {query_cache.key_for(query, k)}")
print(f"Contains before search: {query_cache.contains(query, k)}")

retriever = BM25Retriever(processed_dir)
service = SearchService(retriever, query_cache=query_cache)

t4 = time.perf_counter()
r1 = service.search(query, k=k)
t5 = time.perf_counter()

t6 = time.perf_counter()
r2 = service.search(query, k=k)
t7 = time.perf_counter()

print(f"Cold query: {(t5 - t4):.7f} s")
print(f"Cached query: {(t7 - t6):.7f} s")
print(f"Query cache hit (r1 == r2): {r1 == r2}")
print(f"Contains after search: {query_cache.contains(query, k)}")
PY
```

Expected behavior:
- The second index load should be nearly instantaneous.
- `idx1 is idx2` should be `True`.
- The same query should return the same result from cache on the second execution.
- The SHA-256 key should be deterministic for the same query and `k` value.










last schema:
```
                    ┌──────────────┐
                    │ data/raw/    │
                    │ vllm-0.10.1  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Indexer    │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      PythonChunker             MarkdownChunker
              │                         │
              └────────────┬────────────┘
                           ▼
                    ┌──────────────┐
                    │ LexicalIndex │
                    │    BM25      │
                    └──────┬───────┘
                           │
                           ▼
                  data/processed/index.json
                           │
                           ▼
                    ┌──────────────┐
                    │  Retriever   │
                    │     BM25     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │SearchService │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │build_context │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Qwen/Qwen3-0.6B │
                  │  Transformers   │
                  └─────────────────┘

```



```
Indexer
  ├── descubre y lee archivos
  ├── selecciona chunker
  ├── genera Chunk[]
  │
  ├── LexicalIndexer
  │     └── construye datos BM25
  │
  └── IndexStorage
        └── guarda/carga JSON
```

program starts
      │
      ▼
load BM25 index
      │
      ▼
load metadata
      │
      ▼
200 queries
      │
      ├── tokenize
      ├── BM25 retrieve
      └── map ids → MinimalSource


```
query
  │
  ▼
BM25
  │
  ▼
top-k Chunk
  │
  ▼
build prompt
  │
  ▼
Qwen
  │
  ▼
plain text answer
  │
  ▼
AnsweredQuestion / MinimalAnswer
  │
  ▼
Pydantic validation
  │
  ▼
model_dump_json()
```
```
src/
├── __main__.py
├── cli.py
│
├── models.py
│
├── indexing/
│   ├── indexer.py
│   ├── python_chunker.py
│   ├── markdown_chunker.py
│   └── storage.py
│
├── retrieval/
│   ├── bm25_retriever.py
│   └── ranking.py
│
├── generation/
│   ├── generator.py
│   └── prompt.py
│
└── pipeline.py
```

```
                    MANDATORY

       ┌───────────────┐
       │   vLLM repo   │
       └───────┬───────┘
               │
       ┌───────▼────────┐
       │ Python/MD      │
       │ chunkers       │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ Chunk metadata  │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │      BM25       │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │    Retriever    │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ Context builder │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ Qwen 0.6B       │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │    Pydantic    │
       └───────┬────────┘
               │
             JSON
```
                     data/raw/
                         │
                         ▼
                    IndexPipeline
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       PythonChunker          MarkdownChunker
             │                       │
             └───────────┬───────────┘
                         ▼
                       Chunk
                         │
                         ▼
                       BM25
                         │
                         ▼
                 data/processed/
                         │
                         │
                 ────────┴────────
                         │
                      SEARCH
                         │
                         ▼
                      BM25
                         │
                         ▼
                     ranking
                         │
                         ▼
                     top-k Chunk
                         │
                         ▼
                  MinimalSource[]
                         │
                         ▼
                ContextBuilder
                         │
                         ▼
                    Qwen 0.6B
                         │
                         ▼
                     Pydantic
                         │
                         ▼
                       JSON


 Módulo 1: ranking.py (La Calculadora Matemática)Este módulo es una calculadora pura. No sabe qué es un archivo, ni qué es Python, ni qué pregunta hizo el usuario. Solo recibe números y aplica una fórmula matemática llamada BM25.BM25 es el algoritmo estándar en la industria para medir cómo de "relevante" es un documento respecto a una palabra. Se basa en tres principios lógicos:IDF (Frecuencia Inversa de Documento): Si una palabra aparece en casi todos los archivos del proyecto (por ejemplo, la palabra import o def en Python), esa palabra no es importante porque no ayuda a filtrar. Si una palabra aparece en muy pocos archivos (por ejemplo, calculate_metrics), es una palabra clave muy valiosa. La función calculate_idf calcula este valor de importancia.Frecuencia del término (TF): Si la palabra que buscas aparece 5 veces en un fragmento de texto, ese fragmento es probablemente más relevante que uno donde solo aparece 1 vez.Penalización por longitud: Si un fragmento de texto tiene 2000 palabras y contiene la palabra buscada 1 vez, es menos importante que un fragmento de solo 10 palabras que también la contiene 1 vez. El fragmento corto va directo al grano.

 