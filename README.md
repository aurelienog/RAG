*This project has been created as part of the 42 curriculum by aunoguei.*

# RAG against the machine

## Table of Contents

- [Description](#description)
- [Instructions](#instructions)
- [Resources](#resources)
- [System Architecture](#system-architecture)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Method](#retrieval-method)
- [Performance Analysis](#performance-analysis)
- [Design decisions](#Design-decisions)
- [Challenges faced](#Challenges-faced)
- [Example usage](#Example-usage)
- [Bonus Features(5)](#Bonus-features)

## Description

This project implements a **Retrieval-Augmented Generation (RAG)** system.

The system follows the classic RAG workflow:

1. **Index** a source tree into structured text/code chunks.
2. **Retrieve** the most relevant chunks for a user question.
3. **Generate** an answer with a local causal language model using only the retrieved context.
4. **Evaluate** retrieval quality with Recall@k against a ground-truth dataset.

The system features dedicated chunking strategies for Python and Markdown/text-like files. Indexes are persisted so that search and answer generation run instantly without rebuilding the lexical index every time.

## Instructions

### Requirements

The project requires Python and the following core libraries:

- **CLI & Validation:** `Python Fire`, `Pydantic`, `tqdm`
- **Math & AI Frameworks:** `NumPy`, `PyTorch`, `Transformers`, `Sentence Transformers`
- **Linting & Typing:** `flake8`, `mypy`

### Installation and Input data

By default, the indexer operates on the following paths:
- **Raw Input Data:** `data/raw/vllm-0.10.1/`
- **Processed Index Output:** `data/processed/`

#### Supported Extensions

The source directory is expected to contain supported source files. The indexer accepts:
```
.py .md .markdown .txt .rst .yaml .yml .toml
.c .cpp .h .hpp .cu
```
Several generated/dependency directories are ignored, including ```.git```, virtual environments, ```__pycache__```, ```build```, ```dist```, ```node_modules``` and related directories.

### Compilation & Setup

The project environment, dependencies, and code quality are managed via **`uv`** and a custom **`Makefile`**.

```bash
# Setup the virtual environment and install all dependencies
make install

# Run strict code linting and type checking (flake8 + mypy)
make lint-strict

# Clean standard Python caches (__pycache__, .mypy_cache)
make clean

# Nuke the local environment (removes the entire .venv)
make fclean
```

### Execution

The system uses a unified **Python Fire CLI wrapper**. Commands can be executed natively via the **`Makefile`** parameters or explicitly via explicit package invocations (`uv run python -m src <cmd>`).

#### 1. Ingestion & Index Pipeline
Splits targeted repos into indexed segments. Pass `--max_chunk_size` to customize character limits.

```bash
# Build standard Lexical BM25 Inverted Indices
make index
# Explicit Fire API:
uv run python -m src index --max_chunk_size=2000

# Build combined Lexical and Semantic Transformer Vectors
make index-semantic
# Explicit Fire API:
uv run python -m src index_semantic --max_chunk_size=2000
```

#### 2. Retrieval Search Operations
Search the persisted index and return ranked source paths and character offsets.

```bash
# Option A: Standard Lexical (BM25 Match)
make search QUERY="What is vLLM?" K=10
# Explicit Fire API:
uv run python -m src search --query="What is vLLM?" --k=10

# Option B: Semantic Query (Deduces context using all-MiniLM-L6-v2 vectors)
make search-semantic QUERY="How to configure cache allocation?" K=10
# Explicit Fire API:
uv run python -m src search_semantic --query="How to configure cache allocation?" --k=10

# Option C: Hybrid RRF Search (Combines lexical and vector metrics)
make search-hybrid QUERY="Attention memory constraint overrides" K=10
# Explicit Fire API:
uv run python -m src search_hybrid --query="Attention memory constraint overrides" --k=10
```

#### 3. Isolated Text Generation
Generates answers through prompt matrices bounded by the scoped codebase documentation chunks.

```bash
# Orchestrate full prompt assembly and local text generation output
make answer QUERY="Explain the incremental index verification logic." K=5
# Explicit Fire API:
uv run python -m src answer --query="Explain the incremental index verification logic." --k=5
```

#### 4. Continuous Evaluation & Validation Datasets
Automates structural tracking metrics using public benchmark suites (`UnansweredQuestions` & `AnsweredQuestions`).

```bash
# Stage 1: Batch search all questions from the dataset schema
make search-dataset K=10
# Explicit Fire API:
uv run python -m src search_dataset --k=10

# Stage 2: Run generation models over the output matrices
make answer-dataset
# Explicit Fire API:
uv run python -m src answer_dataset

# Stage 3: Calculate strict validation precision metrics (Recall@k)
make evaluate
# Explicit Fire API:
uv run python -m src evaluate
```

#### 5. Serving Layer (Microservice API)
Spins up production hosting endpoints to serve remote application components.

```bash
# Spin up the underlying API network node
make api HOST="127.0.0.1" PORT=8000
# Explicit Fire API:
uv run python -m src api --host="127.0.0.1" --port=8000
```

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

AI tools were used as a development aid for specific parts of the project:

- **Architecture and design:** discussing the RAG pipeline and comparing
  lexical, semantic, and hybrid retrieval approaches.
- **Retrieval and indexing review:** checking BM25, RRF, incremental indexing,
  embedding persistence, and edge cases in chunking and CLI behavior.
- **Documentation:** reviewing the README structure and helping refine
  explanations and wording.

Suggestions were reviewed and adapted to the actual implementation; the
repository behavior was checked against the project code. The benchmark
figures below are reported project results, not AI-generated estimates.

## System Architecture

The main pipeline is:


```text
                            INDEXING (offline)

                       ┌─────────────────────────┐
                       │ data/raw/vllm-0.10.1/   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │         Indexer         │
                       └────────────┬────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
        ┌────────────────────┐          ┌─────────────────────────┐
        │   PythonChunker    │          │    MarkdownChunker      │
        │      (.py)         │          │ (other supported files) │
        └──────────┬─────────┘          └────────────┬────────────┘
                   └────────────────┬────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ Chunks                  │
                       │ text + path + offsets   │
                       └────────────┬────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
        ┌────────────────────┐          ┌─────────────────────────┐
        │  LexicalIndexer    │          │ Optional embeddings     │
        │       BM25         │          │ Sentence Transformers   │
        └──────────┬─────────┘          └────────────┬────────────┘
                   └────────────────┬────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │      IndexStorage       │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │    data/processed/      │
                       │ index.json + manifest   │
                       │ optional embeddings     │
                       └─────────────────────────┘

                     RETRIEVAL AND ANSWERING (online)

  data/processed/ ───────┬───────────────────────┐
                         ▼                       ▼
               ┌──────────────────┐   ┌────────────────────┐
               │  BM25Retriever   │   │ SemanticRetriever  │
               └────────┬─────────┘   └──────────┬─────────┘
                        │                        │
                        └──────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ HybridRetriever (RRF)     │
                     │ combines both rankings    │
                     └─────────────┬─────────────┘
                                   │
       BM25 / semantic / hybrid ───┤ (one retriever configured per search)
                                   ▼

  SearchService receives one configured retrieval mode:
    ┌──────────────────────────────────────────────────────┐
    │ BM25Retriever                                        │
    │ SemanticRetriever                                    │
    │ HybridRetriever (BM25 + Semantic, combined with RRF) │
    └───────────────────────────┬──────────────────────────┘
                                ▼
                     ┌───────────────────────────┐
  Question ─────────>│      SearchService        │<── optional QueryCache
                     └─────────────┬─────────────┘
                                   │ top-k chunks
             ┌─────────────────────┴────────────────────┐
             ▼                                          ▼
        SEARCH ONLY                                ANSWER GENERATION
   Search results / sources / results JSON      CLI answer / FastAPI
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │   AnswerService    │
                                                └─────────┬──────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │   build_context    │
                                                │ max 12,000 chars   │
                                                └─────────┬──────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │ Qwen/Qwen3-0.6B    │
                                                │ Transformers       │
                                                └─────────┬──────────┘
                                                          │
                                                          ▼
                                                   Answer + sources refs

  Evaluator compares search results with expected ranges to calculate Recall@k.
  search-dataset ──> SearchService ──> results JSON ──> Evaluator ──> Recall@k
  FastAPI exposes /health and /query; /query currently uses BM25 retrieval.
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

Abstract Syntax Tree example: 
```
tree = ast.parse(source)

Module
│
├── ClassDef: User
│   │
│   ├── FunctionDef: __init__
│   │   └── ...
│   │
│   └── FunctionDef: greet
│       └── ...
│
├── FunctionDef: create_user
│   └── ...
│
└── Assign: x = create_user("John")
```

Python chunking workflow:
```
                    Python file
                        │
                        ▼
              ¿is empty?
                 /          \
               yes            no
               │              │
             []          ¿len <= max?
                            /       \
                          yes         no
                          │           │
                          ▼           ▼
                    python_module   ast.parse()
                                      │
                              ┌───────┴────────┐
                              │                │
                           SyntaxError         OK
                              │                │
                              ▼                ▼
                    python_syntax_fallback   tree.body
                                               │
                                               ▼
                                        each top-level node
                                               │
                                ┌──────────────┼──────────────┐
                                │              │              │
                             ClassDef     FunctionDef      Other
                                │              │              │
                                ▼              ▼              ▼
                         python_class   python_function  python_statement
                                │              │              │
                                └──────────────┴──────────────┘
                                               │
                                      ¿Fit into max_size?
                                          /         \
                                        yes           no
                                        │             │
                                        ▼             ▼
                                     1 Chunk    split_lines()
                                                    │
                                                    ▼
                                           python_large_node
```

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
5. paragraph boundaries (`\n\n`)
6. line boundaries (`\n`)

When structural splitting is insufficient, the implementation falls back to line-based splitting and, for oversized individual lines, fixed-size hard splitting.

This strategy aims to preserve semantic/document structure before sacrificing it for the hard size limit.

### Chunk identity & Hashing

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

### How to obtain the project's actual scores

To reproduce the documentation Recall@k scores, provide the configured source
tree and question/reference datasets, then run:

```bash
make index
make search-dataset K=10
make evaluate
```

The default dataset and result paths are defined in `src/config.py`. Evaluation
matches questions by ID, so search results must use IDs present in the answered
dataset. The evaluator reports Recall@1, Recall@3, Recall@5, and Recall@10.

The examination contained 100 questions for the documentation dataset and 100 questions for the code dataset. All 200 questions had valid student sources.

The required Recall@5 thresholds were 80% for documentation and 50% for code. The system achieved 81.0% on documentation and 58.0% on code, so both retrieval requirements were met.

### Performance interpretation

The results show that the retriever is substantially better at identifying relevant documentation than code at low values of ```k```, while increasing ```k``` improves coverage for both datasets. For documentation, Recall@5 reaches 81.0% and Recall@10 reaches 83.0%, indicating that most relevant sources are already present among the first few retrieved results. Code retrieval has lower early-rank recall, but improves from 35.0% at Recall@1 to 68.0% at Recall@10.

### Benchmark limitations

The benchmark evaluates retrieval quality against a private dataset and therefore does not measure answer-generation quality directly. Recall@k indicates whether the expected source appears in the top-k retrieved results; it does not by itself measure the factual correctness, completeness, or clarity of the final generated answer.

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

## Design decisions

### Structure-aware, source-addressable chunks

Python is split using its AST so that functions and classes stay together when
possible. Markdown and text use heading and paragraph boundaries first. Both
strategies fall back to smaller line-based pieces when needed to enforce the
chunk-size limit. Every chunk retains its file path and character offsets, so
retrieval results can point back to precise source ranges and be compared with
the evaluation dataset.

### Incremental indexing

The indexer stores a SHA-256 hash per source file and reuses chunks from
unchanged files. This avoids repeating parsing and chunking work on every run.
Chunk IDs are based on the file path and character range, while the manifest
tracks the IDs associated with each file.

### Complementary retrieval methods

BM25 is the lightweight default and works well for exact technical terms,
identifiers, and filenames. Semantic retrieval is available for meaning-based
matches, but requires model loading and embedding storage. Hybrid retrieval
combines their rankings with Reciprocal Rank Fusion (RRF), which uses rank
positions rather than assuming BM25 and cosine similarity scores are directly
comparable. Semantic indexing is opt-in to keep the standard indexing path
faster and less resource-intensive.

### Portable persistence and explicit provenance

Chunks and the lexical index are stored as JSON, while dense vectors are stored
as NumPy arrays with a separate chunk-ID mapping. The explicit mapping is
validated when embeddings are loaded, helping prevent vectors from being
associated with the wrong source chunks after index updates.

### Bounded, reproducible generation

The generator uses a local causal language model and a prompt that restricts
answers to retrieved context. Context length and generated-token count are
bounded, and sampling is disabled for repeatable output. The retrieval
benchmark is kept separate from answer quality: Recall@k measures whether
expected source ranges were retrieved, not whether a generated answer is
correct.

## Challenges faced

### Preserving useful boundaries under a hard size limit

Code and documentation have different structures, so one splitting rule would
often separate related material or create oversized chunks. AST-aware Python
chunking and hierarchical Markdown splitting preserve structure in the common
case, with line-based and hard-split fallbacks for large nodes, long lines, or
invalid Python syntax.

### Keeping incremental and semantic indexes aligned

Reusing unchanged chunks while rebuilding part of an index makes stale or
misaligned embedding rows a risk. The implementation persists chunk IDs beside
embedding rows and validates dimensions, uniqueness, and references against
the current chunk index before semantic retrieval.

### Balancing exact matches and semantic matches

BM25 is strong for exact API names but can miss paraphrases; dense retrieval
can capture related meaning but may rank exact technical identifiers less
reliably. RRF combines both candidate rankings without mixing their
incompatible raw score scales. The separate retrieval modes also make it
possible to compare the trade-offs directly.

### Evaluating retrieval without overstating answer quality

The available ground truth identifies relevant source ranges, not ideal
generated answers. Recall@k with file and character-range overlap provides a
repeatable retrieval metric, but answer factuality, completeness, and
faithfulness still need a separate evaluation method.

### Running local models within practical resource limits

Embedding every chunk and generating answers locally cost more time and memory
than lexical indexing and search. Semantic indexing is therefore optional,
and the answer pipeline bounds the context and output size. These controls
make the workflow more practical, though generation can still be slow on
CPU-only machines.


## What could be added with more time

- **Answer-quality evaluation:** Add curated answer references and measures for
  factuality, completeness, and whether each claim is supported by cited
  chunks, alongside human review.
- **API flexibility:** Allow the HTTP service to select BM25, semantic, or
  hybrid retrieval through validated configuration, and add deployment-level
  authentication and resource controls where needed.

## Example usage

All input and output paths are configurable CLI arguments.

1. Build the lexical index
```
uv run python -m src index --max_chunk_size 2000
```

```
uv run python -m src index
Indexing files: 100%|██████████████████| 2202/2202 [00:03<00:00, 566.93file/s]
Tokenizing chunks: 100%|███████████| 40915/40915 [00:06<00:00, 6327.18chunk/s]

=== Indexing statistics ===
Files indexed: 2080
Unchanged files: 0
Modified files: 0
New files: 2202
Deleted files: 0
Total chunks: 40915

Ingestion complete! Indices saved under /goinfre/aunoguei/11111/student/data/processed
```

2. Search with BM25

A single-query search returns ranked source locations, each with its file path and character
span.

```
uv run python -m src search "What models does vLLM support?" --k 5
```
```
c3r5s6% uv run python -m src search "What models does vLLM support?" --k 5
data/raw/vllm-0.10.1/docs/getting_started/installation/cpu.md [9438, 9927]
data/raw/vllm-0.10.1/docs/models/supported_models.md [292, 2004]
data/raw/vllm-0.10.1/docs/models/pooling_models.md [0, 700]
data/raw/vllm-0.10.1/docs/models/supported_models.md [50368, 52119]
data/raw/vllm-0.10.1/examples/offline_inference/qwen3_reranker.py [188, 1081]

```

3. Generate an answer
```
uv run python -m src answer "How to deploy vLLM?"
```
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████| 311/311 [00:00<00:00, 6209.64it/s]
{
  "search_results": [
    {
      "question_id": "ee7dc0af-995e-4342-ae24-f3bc87361308",
      "question": "How to deploy vLLM?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/deployment/frameworks/triton.md",
          "first_character_index": 0,
          "last_character_index": 422
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/deployment/frameworks/modal.md",
          "first_character_index": 0,
          "last_character_index": 277
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/deployment/integrations/kubeai.md",
          "first_character_index": 0,
          "last_character_index": 765
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/deployment/frameworks/bentoml.md",
          "first_character_index": 0,
          "last_character_index": 445
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 27170,
          "last_character_index": 27921
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/deployment/frameworks/chatbox.md",
          "first_character_index": 0,
          "last_character_index": 896
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/design/multiprocessing.md",
          "first_character_index": 0,
          "last_character_index": 607
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/design/plugin_system.md",
          "first_character_index": 0,
          "last_character_index": 948
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/deployment/frameworks/anything-llm.md",
          "first_character_index": 0,
          "last_character_index": 1317
        },
        {
          "file_path": "data/raw/vllm-0.10.1/examples/others/lmcache/README.md",
          "first_character_index": 1888,
          "last_character_index": 2039
        }
      ],
      "answer": "To deploy vLLM, you can use the following methods:\n\n1. **Triton Inference Server**: Use the tutorial to deploy a simple model like `facebook/opt-125m` using vLLM.\n2. **Modal**: Deploy vLLM on a serverless platform like Modal.\n3. **KubeAI**: Deploy vLLM on Kubernetes using the KubeAI platform.\n4. **Ray Serve LLM**: Use Ray Serve LLM for scalable and production-grade deployment.\n5. **Anything LLM**: Deploy vLLM as a full-stack application that converts documents into context for LLMs.\n\nThe specific steps depend on the chosen platform and the model you want to deploy."
    }
  ],
  "k": 10
}

```

4. Search a dataset

The public datasets share file names, so writing every run into
the same folder would overwrite previous results.

```
c3r5s6% uv run python -m src search_dataset \
--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
--k 10 \
--save_directory data/output/search_results/UnansweredQuestions  
Searching: 100%|██████████████████████████████████████████████████████| 100/100 [00:02<00:00, 43.20q/s]
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```

5. Generate answers for an existing result set
```
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

output:

```
c3r5s6% make search-dataset 
uv run python -m src search-dataset --k=10
Searching: 100%|██████████████████████████████| 100/100 [00:02<00:00, 44.90q/s]
Saved student_search_results to /goinfre/aunoguei/RAG/student/data/output/search_results/UnansweredQuestions/dataset_docs_public.json
c3r5s6% make answer-dataset 
uv run python -m src answer-dataset
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████| 311/311 [00:01<00:00, 234.23it/s]
Answering: 100%|████████████████████████████| 100/100 [1:30:29<00:00, 54.29s/q]
Saved student_search_results_and_answer to /goinfre/aunoguei/RAG/student/data/output/search_results/dataset_docs_public.json

```

6. Evaluate

Score with the moulinette
```
./moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000 
Student data is valid: True
Total number of questions: 100
Total number of questions with sources: 100
Total number of questions with student sources: 100

🎯 Evaluation Results
========================================
📊 Questions evaluated: 100
📈 Recall@1: 0.610 (61.0%)
📈 Recall@3: 0.790 (79.0%)
📈 Recall@5: 0.840 (84.0%)
📈 Recall@10: 0.890 (89.0%)
{'recall@1': 0.61, 'recall@3': 0.79, 'recall@5': 0.84, 'recall@10': 0.89}
```

With evaluate command:
```
make evaluate
uv run python -m src evaluate
Evaluation Results
========================================
Recall@1: 0.610 Recall@3: 0.790 Recall@5: 0.840 Recall@10: 0.890
```


Alternatively (BONUS):

1. Build the semantic index
```
uv run python -m src index_semantic
```

```
c3r5s6% make index-semantic 
uv run python -m src index-semantic
Indexing files: 100%|█████████████████████████████████████████| 2198/2198 [00:00<00:00, 28558.67file/s]
Tokenizing chunks: 100%|████████████████████████████████████| 40867/40867 [00:06<00:00, 6376.36chunk/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████| 103/103 [00:00<00:00, 452.89it/s]
Encoding embeddings: 100%|████████████████████████████████████| 40867/40867 [15:18<00:00, 44.48chunk/s]
Ingestion complete! Lexical + semantic indices saved under /goinfre/aunoguei/RAG/student/data/processed
```


2. Search semantically
```
uv run python -m src search_semantic
```
```
c3r5s6% make search-semantic
uv run python -m src search-semantic --query="What is vLLM?" --k=10
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████| 103/103 [00:00<00:00, 6588.78it/s]
data/raw/vllm-0.10.1/tests/tpu/lora/test_lora.py [158, 169]
data/raw/vllm-0.10.1/tests/lora/test_mixtral.py [136, 147]
data/raw/vllm-0.10.1/tests/compile/test_config.py [122, 133]
data/raw/vllm-0.10.1/tests/lora/test_minicpmv_tp.py [123, 134]
data/raw/vllm-0.10.1/tests/lora/test_phi.py [108, 119]
data/raw/vllm-0.10.1/vllm/entrypoints/cli/serve.py [182, 193]
data/raw/vllm-0.10.1/tests/lora/test_llama_tp.py [162, 173]
data/raw/vllm-0.10.1/tests/lora/test_chatglm3_tp.py [108, 119]
data/raw/vllm-0.10.1/tests/lora/test_transformers_model.py [123, 134]
data/raw/vllm-0.10.1/tests/lora/test_quant_model.py [273, 284]
```

3. Search with hybrid retrieval
```
uv run python -m src search_hybrid
```

4. Caching

```
uv run python -m src caching --query="What is vLLM?" --k=10
```

5. Local HTTP API:
```
uv run python -m src api
```

```
(rag-against-the-machine) c3r6s6% uv run python -m src api
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████| 311/311 [00:00<00:00, 7745.55it/s]
Loading weights: 100%|████████████████████| 311/311 [00:00<00:00, 6551.36it/s]
INFO:     Started server process [3327631]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:41468 - "GET / HTTP/1.1" 307 Temporary Redirect
INFO:     127.0.0.1:41468 - "GET /docs HTTP/1.1" 200 OK
INFO:     127.0.0.1:41468 - "GET /openapi.json HTTP/1.1" 200 OK
INFO:     127.0.0.1:41476 - "POST /query HTTP/1.1" 200 OK
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [3327631]
```

## Bonus Features

### 1. Semantic embeddings

Semantic embeddings turn each chunk into a dense vector representation in a shared embedding space. Instead of matching exact words, the retriever compares the query embedding with chunk embeddings using cosine similarity, which captures semantic similarity even when the wording differs.

This is useful for conceptual questions, paraphrases, and cases where the relevant code or documentation uses different wording than the query. The trade-off is that semantic indexing requires an extra embedding pass and more storage, but it often retrieves relevant results that BM25 alone would miss.


#### Demonstration

```
c3r5s6% make index-semantic 
uv run python -m src index-semantic
Indexing files: 100%|█████████████████| 2198/2198 [00:00<00:00, 27701.23file/s]
Tokenizing chunks: 100%|████████████| 40867/40867 [00:06<00:00, 6235.61chunk/s]
modules.json: 100%|███████████████████████████| 349/349 [00:00<00:00, 1.80MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config_sentence_transformers.json: 100%|███████| 116/116 [00:00<00:00, 724kB/s]
README.md: 100%|██████████████████████████| 10.5k/10.5k [00:00<00:00, 32.4MB/s]
sentence_bert_config.json: 100%|█████████████| 53.0/53.0 [00:00<00:00, 325kB/s]
config.json: 100%|████████████████████████████| 612/612 [00:00<00:00, 3.68MB/s]
model.safetensors: downloading bytes: █████████████████████| 85.0MB, 7.72MB/s  
model.safetensors: reconstructing file: 100%|█████| 90.9MB / 90.9MB, 8.59MB/s  
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 6149.13it/s]
tokenizer_config.json: 100%|██████████████████| 350/350 [00:00<00:00, 2.18MB/s]
vocab.txt: 100%|████████████████████████████| 232k/232k [00:00<00:00, 11.6MB/s]
tokenizer.json: 100%|███████████████████████| 466k/466k [00:00<00:00, 39.4MB/s]
special_tokens_map.json: 100%|█████████████████| 112/112 [00:00<00:00, 753kB/s]
config.json: 100%|████████████████████████████| 190/190 [00:00<00:00, 1.07MB/s]
Encoding embeddings: 100%|████████████| 40867/40867 [14:11<00:00, 47.98chunk/s]
Ingestion complete! Lexical + semantic indices saved under /goinfre/aunoguei/RAG/student/data/processed
c3r5s6% make search-semantic 
```
```
uv run python -m src search-semantic --query="What is vLLM?" --k=10
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 8159.51it/s]
data/raw/vllm-0.10.1/tests/tpu/lora/test_lora.py [158, 169]
data/raw/vllm-0.10.1/tests/lora/test_mixtral.py [136, 147]
data/raw/vllm-0.10.1/tests/compile/test_config.py [122, 133]
data/raw/vllm-0.10.1/tests/lora/test_minicpmv_tp.py [123, 134]
data/raw/vllm-0.10.1/tests/lora/test_phi.py [108, 119]
data/raw/vllm-0.10.1/vllm/entrypoints/cli/serve.py [182, 193]
data/raw/vllm-0.10.1/tests/lora/test_llama_tp.py [162, 173]
data/raw/vllm-0.10.1/tests/lora/test_chatglm3_tp.py [108, 119]
data/raw/vllm-0.10.1/tests/lora/test_transformers_model.py [123, 134]
data/raw/vllm-0.10.1/tests/lora/test_quant_model.py [273, 284]

```

### 2. Hybrid retrieval

Hybrid retrieval combines the two complementary signals already described in the project: lexical matching from BM25 and semantic similarity from dense embeddings. The system runs both retrievers independently, gathers their top candidate lists, and then merges them using Reciprocal Rank Fusion (RRF), which ranks items by how highly they appear in each list instead of trying to compare incompatible score scales.

This makes the search more robust in real-world questions: BM25 is strong for exact identifiers, filenames, APIs, and technical terms, while the semantic retriever can capture paraphrases and conceptual matches. By combining both rankings, hybrid retrieval usually improves recall on mixed queries that contain both precise keywords and more natural-language intent.

#### Demonstration

```
make search-hybrid  
uv run python -m src search-hybrid --query="What is vLLM?" --k=10
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6739.47it/s]
data/raw/vllm-0.10.1/examples/offline_inference/automatic_prefix_caching.py [5773, 6504]
data/raw/vllm-0.10.1/tests/tpu/lora/test_lora.py [158, 169]
data/raw/vllm-0.10.1/docs/features/multimodal_inputs.md [689, 2657]
data/raw/vllm-0.10.1/tests/lora/test_mixtral.py [136, 147]
data/raw/vllm-0.10.1/tests/models/language/pooling/test_gritlm.py [6595, 7137]
data/raw/vllm-0.10.1/tests/compile/test_config.py [122, 133]
data/raw/vllm-0.10.1/docs/getting_started/installation/cpu.md [9438, 9927]
data/raw/vllm-0.10.1/tests/lora/test_minicpmv_tp.py [123, 134]
data/raw/vllm-0.10.1/tests/lora/test_transformers_model.py [1667, 2715]
data/raw/vllm-0.10.1/tests/lora/test_phi.py [108, 119]
```

### 3. Incremental indexing

#### File indexing

The indexer maintains a manifest of the files that were processed previously. Each file is identified by a SHA-256 hash of its contents. During a new indexing run, the current files are compared with the previous manifest and classified as:

- Unchanged: the file content is identical, so its existing chunks can be reused.
- Modified: the file content changed, so the file is chunked again.
- New: the file did not exist in the previous manifest, so it is chunked and indexed.
- Deleted: the file is no longer present and its previous chunks are removed from the index.

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
Files indexed: 2200
Unchanged files: 2198
Modified files: 1
New files: 1
Deleted files: 1
Total chunks: 40869
```
In this run, 2,198 of 2,200 files were unchanged, while only one file required modification processing, one new and one deleted files were detected. The final index contained 40869 chunks, illustrating the benefit of incremental processing on a large corpus when most files remain unchanged.

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

For each indexing run, the manifest allows the system to determine the minimum required work.
This design provides both correctness and efficiency: changes are detected deterministically, while unaffected data remains available for reuse.

### 4. Caching

This project includes two complementary cache layers to reduce repeated startup and retrieval overhead.

#### Index cache

The in-memory index cache keeps the loaded BM25 index alive inside the same Python process, so repeated loads of the same processed index do not deserialize the same JSON payload again.

This is especially useful when the application reuses the same dataset across multiple searches in the same runtime.

#### Query cache

The query cache persists retrieval results on disk using a deterministic SHA-256 key derived from the normalized query text and the requested result count. This allows repeated queries to reuse previously computed results instead of re-running the search.

This improves cold-start behavior and reduces latency for repeated same-query lookups.

#### Demonstration

Run the cache demonstration from the CLI:

```
uv run python -m src caching --query="what is rag" --k=5
# Or use the Make target:
make caching QUERY="what is rag" K=5
```

Output:
```
Cold load:   1.58190 s
Cached load: 0.0000099 s
Same object in RAM (idx1 is idx2): True
SHA-256 key: cf68d2d14894877e4bf2c3c9f33690656803c729a26e2dde8c07e0b033d9ef41
Contains before search: False
Cold query: 0.0009454 s
Cached query: 0.0001763 s
Query cache hit (r1 == r2): True
Contains after search: True
```

The command compares cold and cached index loads and query searches. It clears the in-memory index cache and JSON entries in the selected query-cache
directory before measuring; defaults are `data/processed` and `data/query_cache`.

Expected behavior:
- The second index load should be nearly instantaneous.
- `idx1 is idx2` should be `True`.
- The same query should return the same result from cache on the second execution.
- The SHA-256 key should be deterministic for the same query and `k` value.

### 5. Local HTTP API

The API exposes a health check and a question-answering endpoint. It uses the
persisted BM25 index and the local Qwen model; semantic and hybrid retrieval
are available through the CLI, but are not selectable through this API.

Build the lexical index before the first launch. The first API startup loads
the Qwen model and may download it from Hugging Face if it is not cached.

Available endpoints:

- `GET /health` returns `{"status":"ok"}` when the service is running.
- `POST /query` accepts a non-empty `question` and an optional positive `k`
  (default: `10`). The response contains the question, generated answer, and
  source references with file paths and character offsets.
- `GET /docs` opens the interactive Swagger UI. Visiting `/` redirects there.

Example requests:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is vLLM?","k":5}'
```

The query response has this shape:

```json
{
  "question": "What is vLLM?",
  "answer": "<generated answer>",
  "sources": [
    {
      "file_path": "docs/example.md",
      "first_character_index": 0,
      "last_character_index": 500
    }
  ]
}
```

Example terminal output:
```
INFO:     Started server process [1418906]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:55438 - "GET / HTTP/1.1" 307 Temporary Redirect
INFO:     127.0.0.1:55456 - "GET /docs HTTP/1.1" 200 OK
INFO:     127.0.0.1:55470 - "GET /openapi.json HTTP/1.1" 200 OK
INFO:     127.0.0.1:41292 - "POST /query HTTP/1.1" 200 OK
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [1418906]
```
