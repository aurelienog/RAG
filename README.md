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
Queries database storage blocks to extract file boundaries and line offsets.

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

## Design decisions



## Challenges faced

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

4. Local HTTP API:
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

#### 

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

### 4. Caching

This project includes two complementary cache layers to reduce repeated startup and retrieval overhead.

#### Index cache

The in-memory index cache keeps the loaded BM25 index alive inside the same Python process, so repeated loads of the same processed index do not deserialize the same JSON payload again.

This is especially useful when the application reuses the same dataset across multiple searches in the same runtime.

#### Query cache

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

```
Cold load:   1.55436 s
Cached load: 0.0000116 s
Same object in RAM (idx1 is idx2): True
SHA-256 key: 4fddadd021dc0c9f72140c1257502c9e031d494f60dc83f514283ee365397846
Contains before search: False
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6315.06it/s]
Cold query: 0.0221394 s
Cached query: 0.0001498 s
Query cache hit (r1 == r2): True
Contains after search: True
```

### 5. Local HTTP API

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

#### Demonstration





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
