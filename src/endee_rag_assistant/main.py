from __future__ import annotations

import argparse
from pathlib import Path

from endee_rag_assistant.chunking import chunk_markdown
from endee_rag_assistant.config import settings
from endee_rag_assistant.endee_client import EndeeClient
from endee_rag_assistant.embeddings import EmbeddingModel
from endee_rag_assistant.rag import answer_with_rag, build_context


def ingest_docs(docs_dir: Path) -> None:
    model = EmbeddingModel(settings.embedding_model)
    client = EndeeClient(settings.endee_base_url, settings.collection_name)
    client.create_collection()

    all_chunks = []
    for file_path in docs_dir.glob("*.md"):
        all_chunks.extend(chunk_markdown(file_path))

    vectors = model.encode([c.text for c in all_chunks])

    points = []
    for chunk, vector in zip(all_chunks, vectors):
        points.append(
            {
                "id": chunk.id,
                "vector": vector,
                "payload": {
                    "text": chunk.text,
                    "source": chunk.source,
                },
            }
        )

    client.upsert_points(points)
    print(f"Ingested {len(points)} chunks into '{settings.collection_name}'.")


def semantic_search(query: str, limit: int) -> None:
    model = EmbeddingModel(settings.embedding_model)
    client = EndeeClient(settings.endee_base_url, settings.collection_name)
    query_vector = model.encode([query])[0]
    results = client.search(query_vector, limit=limit)
    for idx, item in enumerate(results, start=1):
        payload = item.get("payload", {})
        print(f"{idx}. score={item.get('score'):.4f} | {payload.get('source')}")
        print(payload.get("text", ""))
        print("-" * 80)


def ask(question: str, limit: int) -> None:
    model = EmbeddingModel(settings.embedding_model)
    client = EndeeClient(settings.endee_base_url, settings.collection_name)
    query_vector = model.encode([question])[0]
    results = client.search(query_vector, limit=limit)
    context = build_context(results)
    print(answer_with_rag(question, context, settings.openai_model))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Endee-powered Support RAG Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest markdown files into Endee")
    ingest_parser.add_argument("--docs-dir", default="docs", type=Path)

    search_parser = subparsers.add_parser("search", help="Run semantic search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)

    ask_parser = subparsers.add_parser("ask", help="Run RAG question answering")
    ask_parser.add_argument("question", type=str)
    ask_parser.add_argument("--limit", type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_docs(args.docs_dir)
    elif args.command == "search":
        semantic_search(args.query, args.limit)
    elif args.command == "ask":
        ask(args.question, args.limit)


if __name__ == "__main__":
    main()
