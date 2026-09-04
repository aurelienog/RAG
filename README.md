*This project has been created as part of the 42 curriculum by aunoguei.*

# RAG against the machine

## Description

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
       │    Pydantic     │
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

## Instructions

## System architecture: Describe your RAG pipeline components and how they
interact

## Chunking strategy: Explain your approach to document segmentation

## Retrieval method: Detail the retrieval algorithm and ranking mechanism
RAG against the machine Will you answer my questions?

## Performance analysis: Discuss recall@k scores and system performance

## Design decisions: Explain key implementation choices

## Challenges faced: Document difficulties encountered and solutions

## Example usage: Provide clear examples of running your system


## Resources

- [AWS — What is RAG ?](https://aws.amazon.com/fr/what-is/retrieval-augmented-generation/)  

- [freeCodeCamp — RAG & MCP fundamentals?](https://www.youtube.com/watch?v=I7_WXKhyGms)

- [Python Fire - Using a Fire CLI](https://google.github.io/python-fire/guide/)

## AI Usage

- [Tqdm Python](https://www.datacamp.com/tutorial/tqdm-python)

- [uuid Module](https://www.w3schools.com/python/ref_module_uuid.asp)

- [AST](https://earthly.dev/blog/python-ast/)

- [Chunking strategies guide](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)

AI tools were used as a learning aid during the project.

They were used to:

...