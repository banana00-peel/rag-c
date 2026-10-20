from pathlib import Path

import pytest

from app.config.settings import Settings
from app.domain.models import ChunkRecord, DocumentStatus, ExtractedPage
from app.indexing.service import (
    DocumentStateConflictError,
    IndexingService,
)
from app.indexing.chroma_store import ChromaStore
from app.persistence.database import build_engine, build_session_factory, initialize_database
from app.persistence.repository import DocumentRepository
from app.services.document_service import DocumentService


class FakeIngestion:
    def __init__(self, pages: list[ExtractedPage]):
        self.pages = pages

    def inspect(self, path: str | Path) -> list[ExtractedPage]:
        return self.pages


class FakeChunker:
    def __init__(self, chunks: list[ChunkRecord]):
        self.chunks = chunks

    def chunk(self, **kwargs) -> list[ChunkRecord]:
        return self.chunks


class FakeChroma:
    def __init__(self):
        self.ids: set[str] = set()
        self.deleted_documents: list[str] = []
        self.fail_upsert = False

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        if self.fail_upsert:
            raise RuntimeError("模拟向量服务失败")
        self.ids.update(chunk.id for chunk in chunks)

    def delete_document(self, document_id: str) -> None:
        self.deleted_documents.append(document_id)
        self.ids = {chunk_id for chunk_id in self.ids if not chunk_id.startswith(f"{document_id}:")}

    def count_document(self, document_id: str) -> int:
        return sum(chunk_id.startswith(f"{document_id}:") for chunk_id in self.ids)


class FakeBM25:
    def __init__(self):
        self.chunks: list[ChunkRecord] = []
        self.rebuild_count = 0

    def rebuild(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = list(chunks)
        self.rebuild_count += 1


class FakeEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class RecordingExecutor:
    def __init__(self):
        self.document_ids: list[str] = []

    def submit(self, document_id: str) -> None:
        self.document_ids.append(document_id)


def build_repository(tmp_path: Path) -> DocumentRepository:
    engine = build_engine(f"sqlite:///{tmp_path / 'test.db'}")
    initialize_database(engine)
    return DocumentRepository(build_session_factory(engine))


def build_service(tmp_path: Path, repository: DocumentRepository, chunks: list[ChunkRecord]):
    chroma = FakeChroma()
    bm25 = FakeBM25()
    service = IndexingService(
        settings=Settings(upload_dir=tmp_path),
        documents=repository,
        ingestion=FakeIngestion([ExtractedPage(page_number=1, text="说明书内容")]),
        chunker=FakeChunker(chunks),
        chroma=chroma,
        bm25=bm25,
    )
    return service, chroma, bm25


def create_document(repository: DocumentRepository, tmp_path: Path, document_id: str = "doc-1"):
    source_path = tmp_path / f"{document_id}.pdf"
    source_path.write_bytes(b"pdf")
    return repository.create(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        stored_path=str(source_path),
        sha256=document_id.ljust(64, "0"),
        brand="石头",
        model="G20",
    )


def submit_and_execute(service: IndexingService, document_id: str = "doc-1") -> None:
    service.submit_index(document_id, RecordingExecutor())
    service.execute(document_id)


def test_repository_atomically_claims_indexing(tmp_path: Path):
    repository = build_repository(tmp_path)
    create_document(repository, tmp_path)

    claimed = repository.try_mark_processing("doc-1")
    assert claimed is not None
    assert claimed.status == DocumentStatus.PROCESSING
    assert repository.try_mark_processing("doc-1") is None

    repository.mark_failed("doc-1", "模拟失败")
    retry = repository.try_mark_processing("doc-1")
    assert retry is not None
    assert retry.status == DocumentStatus.PROCESSING


def test_document_metadata_normalizes_model_alias(tmp_path: Path):
    repository = build_repository(tmp_path)
    create_document(repository, tmp_path)
    service = DocumentService(Settings(upload_dir=tmp_path), repository)

    updated = service.update_metadata("doc-1", brand="石头", model="g10spro")

    assert updated.model == "G10S Pro"


def test_failed_index_cleans_search_data_and_can_retry(tmp_path: Path):
    repository = build_repository(tmp_path)
    create_document(repository, tmp_path)
    chunks = [
        ChunkRecord("doc-1:0", "doc-1", "doc-1.pdf", "石头", "G20", 1, 0, "说明书内容")
    ]
    service, chroma, bm25 = build_service(tmp_path, repository, chunks)

    service.submit_index("doc-1", RecordingExecutor())
    chroma.fail_upsert = True
    service.execute("doc-1")

    failed = repository.get_by_id("doc-1")
    assert failed is not None
    assert failed.status == DocumentStatus.FAILED
    assert failed.chunk_count == 0
    assert repository.list_chunks() == []
    assert chroma.ids == set()
    assert bm25.chunks == []

    chroma.fail_upsert = False
    submit_and_execute(service)
    completed = repository.get_by_id("doc-1")
    assert completed is not None
    assert completed.status == DocumentStatus.COMPLETED
    assert completed.chunk_count == 1
    assert len(repository.list_chunks()) == 1


def test_completed_document_can_be_reindexed_and_duplicate_is_rejected(tmp_path: Path):
    repository = build_repository(tmp_path)
    create_document(repository, tmp_path)
    chunks = [
        ChunkRecord("doc-1:0", "doc-1", "doc-1.pdf", "石头", "G20", 1, 0, "说明书内容")
    ]
    service, _, _ = build_service(tmp_path, repository, chunks)

    service.submit_index("doc-1", RecordingExecutor())
    with pytest.raises(DocumentStateConflictError):
        service.submit_index("doc-1", RecordingExecutor())
    service.execute("doc-1")

    reindex = service.submit_index("doc-1", RecordingExecutor())
    assert reindex.status == DocumentStatus.PROCESSING
    service.execute("doc-1")
    assert repository.get_by_id("doc-1").status == DocumentStatus.COMPLETED


def test_delete_document_removes_pdf_database_and_search_indexes(tmp_path: Path):
    repository = build_repository(tmp_path)
    document = create_document(repository, tmp_path)
    chunks = [
        ChunkRecord("doc-1:0", "doc-1", "doc-1.pdf", "石头", "G20", 1, 0, "说明书内容")
    ]
    service, chroma, bm25 = build_service(tmp_path, repository, chunks)

    submit_and_execute(service)
    service.delete_document("doc-1")

    assert repository.get_by_id("doc-1") is None
    assert repository.list_chunks() == []
    assert chroma.ids == set()
    assert bm25.chunks == []
    assert not Path(document.stored_path).exists()


def test_processing_document_cannot_be_deleted(tmp_path: Path):
    repository = build_repository(tmp_path)
    create_document(repository, tmp_path)
    service, _, _ = build_service(tmp_path, repository, [])
    service.submit_index("doc-1", RecordingExecutor())

    with pytest.raises(DocumentStateConflictError):
        service.delete_document("doc-1")


def test_chroma_deletes_only_target_document(tmp_path: Path):
    store = ChromaStore(tmp_path / "chroma", FakeEmbeddingProvider())
    chunks = [
        ChunkRecord("doc-1:0", "doc-1", "doc-1.pdf", "石头", "G20", 1, 0, "内容一"),
        ChunkRecord("doc-2:0", "doc-2", "doc-2.pdf", "石头", "S5", 1, 0, "内容二"),
    ]
    store.upsert(chunks)

    store.delete_document("doc-1")

    assert store.count() == 1
    assert store.count_document("doc-1") == 0
    assert store.count_document("doc-2") == 1
    matches = store.query("内容", top_k=5)
    assert [item["id"] for item in matches] == ["doc-2:0"]

