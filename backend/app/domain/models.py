from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

# 文档的处理状态：等待，处理，完成，失败
class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RefusalCode(StrEnum):
    NO_RECALL = "no_recall"
    LOW_RELEVANCE = "low_relevance"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_CITATION = "invalid_citation"

# PDF 中提取的一页文字
@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str

# 准备写入索引的文本块
@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    document_name: str
    brand: str
    model: str
    page_number: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    document_id: str
    document_name: str
    brand: str
    model: str
    page_number: int
    chunk_index: int
    text: str
    dense_distance: float | None
    bm25_score: float | None
    rrf_score: float
    matched_channels: tuple[str, ...]

    @property
    def dense_score(self) -> float | None:
        if self.dense_distance is None:
            return None
        return 1.0 - self.dense_distance


@dataclass(frozen=True)
class RetrievalChannelHit:
    """单个检索通道的可解释命中结果，主要用于离线评测与问题分析。"""

    id: str
    document_id: str
    document_name: str
    model: str
    page_number: int
    chunk_index: int
    rank: int
    query_variants: tuple[str, ...]
    channel_rrf_score: float
    dense_distance: float | None = None
    bm25_score: float | None = None


@dataclass(frozen=True)
class RetrievalTrace:
    """保留 Dense、BM25 与融合结果，线上接口仍只消费 fused_hits。"""

    dense_hits: tuple[RetrievalChannelHit, ...]
    bm25_hits: tuple[RetrievalChannelHit, ...]
    fused_hits: tuple[RetrievedChunk, ...]


@dataclass(frozen=True)
class ContextBlock:
    chunk_ids: tuple[str, ...]
    document_id: str
    document_name: str
    brand: str
    model: str
    page_start: int
    page_end: int
    text: str
    dense_distance: float | None
    bm25_score: float | None
    rrf_score: float
    matched_channels: tuple[str, ...]

    @property
    def dense_score(self) -> float | None:
        if self.dense_distance is None:
            return None
        return 1.0 - self.dense_distance


@dataclass(frozen=True)
class CitationSource:
    """单次回答内的可信来源映射，不写入持久化存储。"""

    source_id: str
    context: ContextBlock


@dataclass(frozen=True)
class ConversationMessage:
    """供问题改写使用的有界会话消息，不作为知识库事实来源。"""

    id: str
    conversation_id: str
    turn_id: str
    role: Literal["user", "assistant"]
    content: str
    standalone_query: str | None
    model: str | None
    refused: bool | None
    refusal_code: RefusalCode | None
    created_at: datetime

# 文档信息的只读快照
@dataclass(frozen=True)
class DocumentSnapshot:
    id: str
    filename: str
    stored_path: str
    sha256: str
    brand: str
    model: str
    status: DocumentStatus
    page_count: int
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime

