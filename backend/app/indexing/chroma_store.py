from pathlib import Path
from typing import Any

import chromadb

from app.domain.models import ChunkRecord
from app.indexing.embeddings import EmbeddingProvider


class ChromaStore:
    def __init__(self, path: str | Path, embedding_provider: EmbeddingProvider):
        self._embedding_provider = embedding_provider
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name="roborock_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        embeddings = self._embedding_provider.embed_documents([chunk.text for chunk in chunks])
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self._collection.count()

    def count_document(self, document_id: str) -> int:
        result = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        return len(result.get("ids", []))

    def query(
        self,
        query: str,
        *,
        top_k: int,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        where = {"model": model} if model and model != "all" else None
        query_embedding = self._embedding_provider.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "id": ids[index],
                "text": documents[index],
                "metadata": metadatas[index] or {},
                "distance": float(distances[index]) if distances[index] is not None else None,
            }
            for index in range(len(ids))
        ]

    @staticmethod
    def _metadata(chunk: ChunkRecord) -> dict[str, str | int]:
        return {
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "brand": chunk.brand,
            "model": chunk.model,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }

