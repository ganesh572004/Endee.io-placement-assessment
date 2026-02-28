from __future__ import annotations


def build_context(results: list[dict], max_chunks: int = 4) -> str:
    chosen = results[:max_chunks]
    lines = []
    for i, item in enumerate(chosen, start=1):
        payload = item.get("payload", {})
        lines.append(f"[{i}] {payload.get('text', '')} (source: {payload.get('source', 'unknown')})")
    return "\n".join(lines)


def answer_with_rag(question: str, context: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "You are a support assistant. Answer using only the provided context. If unsure, say so.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return completion.choices[0].message.content or "No answer generated."
