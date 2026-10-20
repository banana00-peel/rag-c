from pathlib import Path

import pytest

from app.config.settings import Settings
from app.domain.models import ChunkRecord, DocumentStatus, ExtractedPage
from app.indexing.service import (
    DocumentStateConflictError,
    IndexingService,
    IndexTaskSubmissionError,
)
from app.persistence.database import build_engine, build_session_factory, initialize_database
from app.persistence.repository import DocumentRepository


class StaticIngestion:
    def inspect(self, path: str | Path) -> list[ExtractedPage]:
        return [ExtractedPage(page_number=1, text="说明书内容")]


class StaticChunker:
    def __init__(self, chunks: list[ChunkRecord]):
        self._chunks = chunks

    def chunk(self, **kwargs) -> list[ChunkRecord]:
        return self._chunks


class ControllableChroma:
    def __init__(self):
        self.ids: set[str] = set()
        self.count_offset = 0
        self.upsert_error: Exception | None = None

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        if self.upsert_error is not None:
            raise self.upsert_error
        self.ids.update(chunk.id for chunk in chunks)

    def delete_document(self, document_id: str) -> None:
        self.ids = {
            chunk_id
            for chunk_id in self.ids
            if not chunk_id.startswith(f"{document_id}:")
        }

    def count_document(self, document_id: str) -> int:
        count = sum(chunk_id.startswith(f"{document_id}:") for chunk_id in self.ids)
        return count + self.count_offset


class TrackingBM25:
    def __init__(self):
        self.chunks: list[ChunkRecord] = []
        self.rebuild_count = 0

    def rebuild(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = list(chunks)
        self.rebuild_count += 1


class RecordingExecutor:
    def __init__(self, error: Exception | None = None):
        self.document_ids: list[str] = []
        self._error = error

    def submit(self, document_id: str) -> None:
        if self._error is not None:
            raise self._error
        self.document_ids.append(document_id)


def build_repository(tmp_path: Path) -> DocumentRepository:
    engine = build_engine(f"sqlite:///{tmp_path / 'reliability.db'}")
    initialize_database(engine)
    return DocumentRepository(build_session_factory(engine))


def create_document(repository: DocumentRepository, tmp_path: Path) -> None:
    source_path = tmp_path / "doc-1.pdf"
    source_path.write_bytes(b"pdf")
    repository.create(
        document_id="doc-1",
        filename="doc-1.pdf",
        stored_path=str(source_path),
        sha256="1" * 64,
        brand="石头",
        model="G20",
    )


def build_service(tmp_path: Path):
    repository = build_repository(tmp_path)
    create_document(repository, tmp_path)
    chunks = [
        ChunkRecord(
            "doc-1:0",
            "doc-1",
            "doc-1.pdf",
            "石头",
            "G20",
            1,
            0,
            "说明书内容",
        )
    ]
    chroma = ControllableChroma()
    bm25 = TrackingBM25()
    service = IndexingService(
        settings=Settings(upload_dir=tmp_path),
        documents=repository,
        ingestion=StaticIngestion(),
        chunker=StaticChunker(chunks),
        chroma=chroma,
        bm25=bm25,
    )
    return service, repository, chroma, bm25, chunks


def test_submit_index_only_enqueues_once(tmp_path: Path):
    service, _, _, _, _ = build_service(tmp_path)
    executor = RecordingExecutor()

    claimed = service.submit_index("doc-1", executor)
    with pytest.raises(DocumentStateConflictError):
        service.submit_index("doc-1", executor)

    assert claimed.status == DocumentStatus.PROCESSING
    assert executor.document_ids == ["doc-1"]


def test_executor_submission_failure_is_compensated_to_failed(tmp_path: Path):
    service, repository, _, _, _ = build_service(tmp_path)
    executor = RecordingExecutor(RuntimeError("executor unavailable"))

    with pytest.raises(IndexTaskSubmissionError):
        service.submit_index("doc-1", executor)

    document = repository.get_by_id("doc-1")
    assert document is not None
    assert document.status == DocumentStatus.FAILED
    assert document.error_message == "索引任务提交失败，请稍后重新提交。"


def test_recovery_cleans_interrupted_task_and_rebuilds_bm25_once(tmp_path: Path):
    service, repository, chroma, bm25, chunks = build_service(tmp_path)
    repository.try_mark_processing("doc-1")
    repository.replace_chunks("doc-1", chunks)
    chroma.upsert(chunks)

    service.recover_interrupted_tasks()

    document = repository.get_by_id("doc-1")
    assert document is not None
    assert document.status == DocumentStatus.FAILED
    assert document.error_message == (
        "服务重启导致索引任务中断，已清理不完整索引，请重新提交。"
    )
    assert repository.count_chunks("doc-1") == 0
    assert chroma.ids == set()
    assert bm25.chunks == []
    assert bm25.rebuild_count == 1


def test_successful_index_passes_sqlite_and_chroma_count_check(tmp_path: Path):
    service, repository, chroma, _, _ = build_service(tmp_path)
    service.submit_index("doc-1", RecordingExecutor())

    service.execute("doc-1")

    document = repository.get_by_id("doc-1")
    assert document is not None
    assert document.status == DocumentStatus.COMPLETED
    assert document.chunk_count == 1
    assert repository.count_chunks("doc-1") == 1
    assert chroma.count_document("doc-1") == 1


def test_count_mismatch_marks_failed_and_cleans_both_stores(tmp_path: Path):
    service, repository, chroma, bm25, _ = build_service(tmp_path)
    chroma.count_offset = 1
    service.submit_index("doc-1", RecordingExecutor())

    service.execute("doc-1")

    document = repository.get_by_id("doc-1")
    assert document is not None
    assert document.status == DocumentStatus.FAILED
    assert document.error_message == (
        "索引一致性校验失败，已清理不完整索引，请重新提交。"
    )
    assert repository.count_chunks("doc-1") == 0
    assert chroma.ids == set()
    assert bm25.chunks == []


def test_upstream_english_error_is_not_persisted(tmp_path: Path):
    service, repository, chroma, _, _ = build_service(tmp_path)
    chroma.upsert_error = RuntimeError("upstream request timeout")
    service.submit_index("doc-1", RecordingExecutor())

    service.execute("doc-1")

    document = repository.get_by_id("doc-1")
    assert document is not None
    assert document.status == DocumentStatus.FAILED
    assert document.error_message == (
        "向量索引写入失败，请检查 Embedding 服务配置和网络连接。"
    )
    assert "upstream" not in document.error_message


def test_terminal_status_only_accepts_processing_documents(tmp_path: Path):
    _, repository, _, _, _ = build_service(tmp_path)

    with pytest.raises(RuntimeError):
        repository.mark_completed("doc-1", page_count=1, chunk_count=1)
    with pytest.raises(RuntimeError):
        repository.mark_failed("doc-1", "不应写入")

    assert repository.get_by_id("doc-1").status == DocumentStatus.PENDING
