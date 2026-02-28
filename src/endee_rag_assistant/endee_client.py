from __future__ import annotations

from typing import Any
import requests


class EndeeClient:
    """REST client using Qdrant-compatible vector APIs exposed by Endee."""

    def __init__(self, base_url: str, collection_name: str, vector_size: int = 384) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.vector_size = vector_size

    def create_collection(self) -> None:
        payload = {
            "vectors": {
                "size": self.vector_size,
                "distance": "Cosine",
            }
        }
        self._request("PUT", f"/collections/{self.collection_name}", json=payload)

    def upsert_points(self, points: list[dict[str, Any]]) -> None:
        payload = {"points": points}
        self._request("PUT", f"/collections/{self.collection_name}/points?wait=true", json=payload)

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        payload = {"vector": query_vector, "limit": limit, "with_payload": True}
        response = self._request("POST", f"/collections/{self.collection_name}/points/search", json=payload)
        return response.get("result", [])

    def _request(self, method: str, path: str, json: dict[str, Any]) -> dict[str, Any]:
        response = requests.request(method=method, url=f"{self.base_url}{path}", json=json, timeout=30)
        response.raise_for_status()
        return response.json()
