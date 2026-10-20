from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update

from app.domain.models import ChunkRecord, DocumentSnapshot, DocumentStatus
from app.persistence.database import ChunkEntity, DocumentEntity


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

# 封装数据库操作
class DocumentRepository:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def create(
        self,
        *,
        document_id: str,
        filename: str,
        stored_path: str,
        sha256: str,
        brand: str,
        model: str,
    ) -> DocumentSnapshot:
        with self._session_factory() as session:
            entity = DocumentEntity(
                id=document_id,
                filename=filename,
                stored_path=stored_path,
                sha256=sha256,
                brand=brand,
                model=model,
                status=DocumentStatus.PENDING.value,
            )
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return self._to_snapshot(entity)

    def get_by_id(self, document_id: str) -> DocumentSnapshot | None:
        with self._session_factory() as session:
            entity = session.get(DocumentEntity, document_id)
            return self._to_snapshot(entity) if entity else None

    def get_by_sha256(self, sha256: str) -> DocumentSnapshot | None:
        with self._session_factory() as session:
            entity = session.scalar(select(DocumentEntity).where(DocumentEntity.sha256 == sha256))
            return self._to_snapshot(entity) if entity else None

    def list_all(self) -> list[DocumentSnapshot]:
        with self._session_factory() as session:
            entities = session.scalars(
                select(DocumentEntity).order_by(DocumentEntity.created_at.desc())
            ).all()
            return [self._to_snapshot(entity) for entity in entities]

    def list_by_status(self, status: DocumentStatus) -> list[DocumentSnapshot]:
        with self._session_factory() as session:
            entities = session.scalars(
                select(DocumentEntity)
                .where(DocumentEntity.status == status.value)
                .order_by(DocumentEntity.created_at)
            ).all()
            return [self._to_snapshot(entity) for entity in entities]

    def update_metadata(self, document_id: str, *, brand: str, model: str) -> DocumentSnapshot:
        with self._session_factory() as session:
            entity = session.get(DocumentEntity, document_id)
            if entity is None:
                raise KeyError(document_id)
            entity.brand = brand.strip()
            entity.model = model.strip()
            entity.updated_at = _utc_now()
            session.commit()
            session.refresh(entity)
            return self._to_snapshot(entity)

    def try_mark_processing(self, document_id: str) -> DocumentSnapshot | None:
        """原子抢占索引执行权，避免同一文档被重复提交。"""
        allowed_statuses = (
            DocumentStatus.PENDING.value,
            DocumentStatus.FAILED.value,
            DocumentStatus.COMPLETED.value,
        )
        with self._session_factory() as session:
            result = session.execute(
                update(DocumentEntity)
                .where(
                    DocumentEntity.id == document_id,
                    DocumentEntity.status.in_(allowed_statuses),
                )
                .values(
                    status=DocumentStatus.PROCESSING.value,
                    error_message=None,
                    updated_at=_utc_now(),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                return None
            session.commit()
            entity = session.get(DocumentEntity, document_id)
            if entity is None:
                return None
            return self._to_snapshot(entity)

    def mark_completed(
        self,
        document_id: str,
        *,
        page_count: int,
        chunk_count: int,
    ) -> DocumentSnapshot:
        with self._session_factory() as session:
            result = session.execute(
                update(DocumentEntity)
                .where(
                    DocumentEntity.id == document_id,
                    DocumentEntity.status == DocumentStatus.PROCESSING.value,
                )
                .values(
                    status=DocumentStatus.COMPLETED.value,
                    page_count=page_count,
                    chunk_count=chunk_count,
                    error_message=None,
                    updated_at=_utc_now(),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                self._raise_invalid_transition(session, document_id)
            session.commit()
            entity = session.get(DocumentEntity, document_id)
            return self._to_snapshot(entity)

    def mark_failed(self, document_id: str, error_message: str) -> DocumentSnapshot:
        with self._session_factory() as session:
            result = session.execute(
                update(DocumentEntity)
                .where(
                    DocumentEntity.id == document_id,
                    DocumentEntity.status == DocumentStatus.PROCESSING.value,
                )
                .values(
                    status=DocumentStatus.FAILED.value,
                    page_count=0,
                    chunk_count=0,
                    error_message=error_message[:4000],
                    updated_at=_utc_now(),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                self._raise_invalid_transition(session, document_id)
            session.commit()
            entity = session.get(DocumentEntity, document_id)
            return self._to_snapshot(entity)

    def replace_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        with self._session_factory() as session:
            session.execute(delete(ChunkEntity).where(ChunkEntity.document_id == document_id))
            session.add_all(
                [
                    ChunkEntity(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        brand=chunk.brand,
                        model=chunk.model,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                    )
                    for chunk in chunks
                ]
            )
            session.commit()

    def delete_chunks(self, document_id: str) -> None:
        with self._session_factory() as session:
            session.execute(delete(ChunkEntity).where(ChunkEntity.document_id == document_id))
            session.commit()

    def count_chunks(self, document_id: str) -> int:
        with self._session_factory() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(ChunkEntity)
                    .where(ChunkEntity.document_id == document_id)
                )
                or 0
            )

    def delete_document(self, document_id: str) -> bool:
        with self._session_factory() as session:
            session.execute(delete(ChunkEntity).where(ChunkEntity.document_id == document_id))
            result = session.execute(
                delete(DocumentEntity).where(DocumentEntity.id == document_id)
            )
            session.commit()
            return result.rowcount == 1

    def list_chunks(self) -> list[ChunkRecord]:
        with self._session_factory() as session:
            entities = session.scalars(
                select(ChunkEntity).order_by(ChunkEntity.document_id, ChunkEntity.chunk_index)
            ).all()
            return [
                ChunkRecord(
                    id=entity.id,
                    document_id=entity.document_id,
                    document_name=entity.document_name,
                    brand=entity.brand,
                    model=entity.model,
                    page_number=entity.page_number,
                    chunk_index=entity.chunk_index,
                    text=entity.text,
                )
                for entity in entities
            ]

    def count_completed(self) -> int:
        with self._session_factory() as session:
            return len(
                session.scalars(
                    select(DocumentEntity).where(
                        DocumentEntity.status == DocumentStatus.COMPLETED.value
                    )
                ).all()
            )

    @staticmethod
    def _raise_invalid_transition(session, document_id: str) -> None:
        entity = session.get(DocumentEntity, document_id)
        if entity is None:
            raise KeyError(document_id)
        raise RuntimeError(
            f"文档状态已变更，不能从 {entity.status} 完成当前索引状态流转。"
        )

    @staticmethod
    def _to_snapshot(entity: DocumentEntity) -> DocumentSnapshot:
        return DocumentSnapshot(
            id=entity.id,
            filename=entity.filename,
            stored_path=entity.stored_path,
            sha256=entity.sha256,
            brand=entity.brand,
            model=entity.model,
            status=DocumentStatus(entity.status),
            page_count=entity.page_count,
            chunk_count=entity.chunk_count,
            error_message=entity.error_message,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

