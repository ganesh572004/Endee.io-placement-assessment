# Endee Support Copilot (RAG + Semantic Search)

A practical AI/ML project that uses **Endee** as the vector database to power:
1. **Semantic search** over support documents.
2. **RAG (Retrieval-Augmented Generation)** for grounded Q&A.

---

## 1) Project overview and problem statement

Support teams lose time searching long policy docs, troubleshooting guides, and plan details.
This project solves that by indexing support knowledge into Endee and enabling:

- natural-language semantic retrieval (`endee-rag search "How long do refunds take?"`)
- grounded answers (`endee-rag ask "Can enterprise users get custom SLAs?"`)

The result is faster support resolution and more consistent responses.

---

## 2) System design and technical approach

### Architecture

1. **Ingestion pipeline**
   - Load Markdown documents from `docs/`.
   - Chunk text with overlap to preserve context.
   - Generate dense embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
   - Upsert vectors + metadata (text/source) into an Endee collection.

2. **Semantic retrieval**
   - Embed the user query.
   - Execute vector similarity search in Endee.
   - Return top-k chunks.

3. **RAG answer generation**
   - Build prompt context from retrieved chunks.
   - Ask an LLM (OpenAI API) to answer strictly from retrieved evidence.

### Why this design

- Endee handles vector indexing/search efficiently.
- Local chunking + embeddings keep ingestion simple and transparent.
- RAG produces grounded, source-linked answers instead of hallucinated free-form responses.

---

## 3) How Endee is used

Endee is the **core vector database** in this project:

- Collection management (`create_collection`)
- Vector point ingestion (`upsert_points`)
- Similarity retrieval (`search`)

The adapter is implemented in:
- `src/endee_rag_assistant/endee_client.py`

> Note: The client uses a Qdrant-compatible REST shape that Endee exposes in common setups. If your Endee deployment uses different endpoint paths/auth headers, update `EndeeClient` accordingly.

---

## 4) Repository usage steps (mandatory for evaluation)

Before starting implementation, perform these required steps:

1. Star the official Endee repository: `https://github.com/endee-io/endee`
2. Fork that repository into your personal GitHub account.
3. Use your fork as the base repository for your project work.

This preserves uniformity and authenticity for assessment.

---

## 5) Setup and execution instructions

## Prerequisites

- Python 3.10+
- Running Endee instance reachable via HTTP
- (Optional for `ask`) OpenAI API key

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Environment variables

```bash
export ENDEE_BASE_URL=http://localhost:6333
export ENDEE_COLLECTION=support_knowledge
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export OPENAI_API_KEY=your_key_here            # only needed for RAG answer generation
export OPENAI_MODEL=gpt-4o-mini
```

### Ingest documents into Endee

```bash
endee-rag ingest --docs-dir docs
```

### Run semantic search

```bash
endee-rag search "How long do refunds take?" --limit 5
```

### Run RAG Q&A

```bash
endee-rag ask "Do enterprise plans include SSO?" --limit 5
```

---

## 6) Project structure

```text
.
├── docs/
│   └── product_docs.md
├── src/endee_rag_assistant/
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── endee_client.py
│   ├── main.py
│   └── rag.py
├── tests/
│   ├── test_chunking.py
│   └── test_rag.py
├── pyproject.toml
└── README.md
```

---

## 7) Future enhancements

- Add metadata filters (product area, language, version)
- Add reranking model for retrieval quality
- Add feedback logging and relevance metrics (MRR/NDCG)
- Add web UI and conversation memory for agentic workflows

