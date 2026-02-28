from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    text: str
    source: str


def chunk_markdown(file_path: Path, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    content = file_path.read_text(encoding="utf-8")
    normalized = " ".join(content.split())
    chunks: list[Chunk] = []

    start = 0
    idx = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        text = normalized[start:end]
        chunks.append(Chunk(id=f"{file_path.stem}-{idx}", text=text, source=str(file_path)))
        idx += 1
        if end == len(normalized):
            break
        start = max(0, end - overlap)

    return chunks
