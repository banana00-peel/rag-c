from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Roborock RAG"
    app_env: str = "development"
    api_prefix: str = "/api"
    backend_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"

    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/documents")
    chroma_dir: Path = Path("data/chroma")
    bm25_snapshot_path: Path = Path("data/bm25/snapshot.json")
    database_url: str = "sqlite:///./data/app.db"

    max_upload_mb: int = Field(default=50, ge=1, le=500)
    chunk_strategy: Literal["fixed", "overlap", "recursive"] = "recursive"
    chunk_size: int = Field(default=800, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)

    dense_top_k: int = Field(default=8, ge=1, le=50)
    bm25_top_k: int = Field(default=8, ge=1, le=50)
    context_top_k: int = Field(default=5, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    max_dense_distance: float = Field(default=0.75, ge=0.0, le=2.0)
    min_bm25_score: float = Field(default=0.0, ge=0.0)
    max_context_chunks_per_document: int = Field(default=3, ge=1, le=20)
    context_merge_max_chars: int = Field(default=1600, ge=200, le=10000)

    conversation_history_turns: int = Field(default=6, ge=1, le=20)
    conversation_history_max_chars: int = Field(default=6000, ge=500, le=30000)
    conversation_ttl_minutes: int = Field(default=60, ge=5, le=1440)

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_snapshot_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

