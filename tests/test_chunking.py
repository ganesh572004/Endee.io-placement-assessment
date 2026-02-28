from pathlib import Path

from endee_rag_assistant.chunking import chunk_markdown


def test_chunk_markdown_creates_multiple_chunks(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text("word " * 400, encoding="utf-8")

    chunks = chunk_markdown(file_path, chunk_size=200, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].id == "sample-0"
    assert all(chunk.source.endswith("sample.md") for chunk in chunks)
