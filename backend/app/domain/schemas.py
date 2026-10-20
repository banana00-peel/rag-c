from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import DocumentStatus, RefusalCode


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    brand: str
    model: str
    status: DocumentStatus
    page_count: int
    chunk_count: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    document: DocumentResponse
    duplicate: bool = False


class DocumentMetadataUpdate(BaseModel):
    brand: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(default=None, max_length=100)
    session_id: UUID | None = None


class SourceResponse(BaseModel):
    source_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    model: str
    snippet: str
    chunk_ids: list[str]
    dense_score: float | None = None
    dense_distance: float | None = None
    bm25_score: float | None = None
    rrf_score: float


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    refusal_code: RefusalCode | None = None
    refusal_reason: str | None = None
    sources: list[SourceResponse] = Field(default_factory=list)
    session_id: UUID
    standalone_query: str
    history_used: bool


class HealthResponse(BaseModel):
    status: str
    deepseek_configured: bool
    embedding_configured: bool
    indexed_documents: int

