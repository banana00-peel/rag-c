from collections.abc import Sequence
from typing import Protocol

from langchain_openai import OpenAIEmbeddings

from app.config.settings import Settings


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, settings: Settings):
        self._client = OpenAIEmbeddings(
            api_key=settings.embedding_api_key or "embedding-key-not-configured",
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            max_retries=2,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

