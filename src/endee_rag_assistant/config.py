from dataclasses import dataclass
import os


@dataclass
class Settings:
    endee_base_url: str = os.getenv("ENDEE_BASE_URL", "http://localhost:6333")
    collection_name: str = os.getenv("ENDEE_COLLECTION", "support_knowledge")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
