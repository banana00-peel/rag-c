import logging
from pathlib import Path
from threading import RLock

from app.config.settings import Settings
from app.domain.models import DocumentSnapshot, DocumentStatus
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pdf import PdfExtractionError
from app.ingestion.service import PdfIngestionService
from app.indexing.bm25_store import BM25Store
from app.indexing.chroma_store import ChromaStore
from app.indexing.task_executor import IndexTaskExecutor
from app.persistence.repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentNotFoundError(LookupError):
    pass


class DocumentStateConflictError(RuntimeError):
    pass


class DocumentDeletionError(RuntimeError):
    pass


class IndexTaskSubmissionError(RuntimeError):
    pass


class IndexRecoveryError(RuntimeError):
    pass


class _IndexingStageError(RuntimeError):
    """携带可以安全展示并持久化给用户的中文错误。"""


_INTERRUPTED_ERROR_MESSAGE = "服务重启导致索引任务中断，已清理不完整索引，请重新提交。"
_UNKNOWN_ERROR_MESSAGE = "索引任务执行失败，请查看服务日志后重试。"


class IndexingService:
    def __init__(
        self,
        *,
        settings: Settings,
        documents: DocumentRepository,
        ingestion: PdfIngestionService,
        chunker: DocumentChunker,
        chroma: ChromaStore,
        bm25: BM25Store,
    ):
        self._settings = settings
        self._documents = documents
        self._ingestion = ingestion
        self._chunker = chunker
        self._chroma = chroma
        self._bm25 = bm25
        self._write_lock = RLock()


    def submit_index(
        self,
        document_id: str,
        executor: IndexTaskExecutor,
    ) -> DocumentSnapshot:
        """原子抢占执行权并通过任务执行器提交，避免重复索引。"""
        current = self._documents.get_by_id(document_id)
        if current is None:
            raise DocumentNotFoundError(document_id)
        if current.status == DocumentStatus.PROCESSING:
            raise DocumentStateConflictError("文档正在索引，请勿重复提交。")

        with self._write_lock:
            document = self._documents.get_by_id(document_id)
            if document is None:
                raise DocumentNotFoundError(document_id)
            if not document.model.strip():
                raise DocumentStateConflictError("请先填写文档型号，再开始索引。")

            claimed = self._documents.try_mark_processing(document_id)
            if claimed is None:
                raise DocumentStateConflictError("文档正在索引，请勿重复提交。")

            try:
                executor.submit(document_id)
            except Exception as exc:
                logger.exception("索引任务提交失败 document_id=%s", document_id)
                try:
                    self._documents.mark_failed(
                        document_id,
                        "索引任务提交失败，请稍后重新提交。",
                    )
                except Exception:
                    logger.exception("补偿索引任务提交状态失败 document_id=%s", document_id)
                raise IndexTaskSubmissionError("索引任务提交失败，请稍后重试。") from exc
            return claimed


    def execute(self, document_id: str) -> None:
        """执行已抢占的索引任务；该入口可由不同任务执行器调用。"""
        with self._write_lock:
            document = self._documents.get_by_id(document_id)
            if document is None:
                logger.error("索引失败：文档不存在，document_id=%s", document_id)
                return
            if document.status != DocumentStatus.PROCESSING:
                logger.warning(
                    "忽略未获得执行权的索引任务，document_id=%s status=%s",
                    document_id,
                    document.status,
                )
                return

            try:
                try:
                    pages = self._ingestion.inspect(document.stored_path)
                except PdfExtractionError as exc:
                    raise _IndexingStageError(self._pdf_error_message(exc)) from exc
                except Exception as exc:
                    raise _IndexingStageError(
                        "文档解析失败，请确认 PDF 可以正常读取。"
                    ) from exc

                if not document.model:
                    raise _IndexingStageError("无法识别型号，请先在上传页面填写型号。")

                try:
                    chunks = self._chunker.chunk(
                        document_id=document.id,
                        document_name=document.filename,
                        brand=document.brand,
                        model=document.model,
                        pages=pages,
                    )
                except Exception as exc:
                    raise _IndexingStageError("文档分块失败，请重新提交索引。") from exc
                if not chunks:
                    raise _IndexingStageError("文档没有可建立索引的文本分块。")

                try:
                    self._chroma.delete_document(document.id)
                    self._chroma.upsert(chunks)
                except Exception as exc:
                    raise _IndexingStageError(
                        "向量索引写入失败，请检查 Embedding 服务配置和网络连接。"
                    ) from exc

                try:
                    self._documents.replace_chunks(document.id, chunks)
                except Exception as exc:
                    raise _IndexingStageError("分块数据保存失败，请重新提交索引。") from exc

                self._verify_index_consistency(document.id, len(chunks))

                try:
                    self._bm25.rebuild(self._documents.list_chunks())
                except Exception as exc:
                    raise _IndexingStageError("关键词索引重建失败，请重新提交索引。") from exc

                self._documents.mark_completed(
                    document.id,
                    page_count=len(pages),
                    chunk_count=len(chunks),
                )
                logger.info(
                    "索引完成 document=%s pages=%d chunks=%d model=%s",
                    document.filename,
                    len(pages),
                    len(chunks),
                    document.model,
                )
            except Exception as exc:
                logger.exception("索引失败 document_id=%s", document_id)
                self._clear_document_index(document_id)
                try:
                    error_message = (
                        str(exc) if isinstance(exc, _IndexingStageError) else _UNKNOWN_ERROR_MESSAGE
                    )
                    self._documents.mark_failed(document_id, error_message)
                except (KeyError, RuntimeError):
                    logger.exception("记录索引失败状态时状态已变化，document_id=%s", document_id)


    def recover_interrupted_tasks(self) -> None:
        """清理进程退出时遗留的处理中任务，并初始化 BM25 索引。"""
        with self._write_lock:
            interrupted = self._documents.list_by_status(DocumentStatus.PROCESSING)
            recovery_errors: list[Exception] = []

            for document in interrupted:
                cleaned = True
                try:
                    self._chroma.delete_document(document.id)
                except Exception as exc:
                    cleaned = False
                    recovery_errors.append(exc)
                    logger.exception("恢复时清理 Chroma 失败 document_id=%s", document.id)

                try:
                    self._documents.delete_chunks(document.id)
                except Exception as exc:
                    cleaned = False
                    recovery_errors.append(exc)
                    logger.exception("恢复时清理 SQLite 分块失败 document_id=%s", document.id)

                if cleaned:
                    try:
                        self._documents.mark_failed(document.id, _INTERRUPTED_ERROR_MESSAGE)
                    except Exception as exc:
                        recovery_errors.append(exc)
                        logger.exception("恢复时更新文档状态失败 document_id=%s", document.id)

            try:
                self._bm25.rebuild(self._documents.list_chunks())
            except Exception as exc:
                recovery_errors.append(exc)
                logger.exception("应用启动时重建 BM25 失败")

            if recovery_errors:
                raise IndexRecoveryError(
                    "索引任务恢复失败，为避免使用不一致的知识库，应用已停止启动。"
                ) from recovery_errors[0]

            if interrupted:
                logger.warning("已恢复 %d 个中断的索引任务", len(interrupted))

    # 删除文档的向量，片段
    def delete_document(self, document_id: str) -> None:
        document = self._documents.get_by_id(document_id)
        if document is not None and document.status == DocumentStatus.PROCESSING:
            raise DocumentStateConflictError("文档正在索引，暂时不能删除。")

        with self._write_lock:
            document = self._documents.get_by_id(document_id)
            if document is not None and document.status == DocumentStatus.PROCESSING:
                raise DocumentStateConflictError("文档正在索引，暂时不能删除。")

            source_path = (
                Path(document.stored_path)
                if document is not None
                else self._settings.upload_dir / f"{document_id}.pdf"
            )
            try:
                self._chroma.delete_document(document_id)
            except Exception as exc:
                logger.exception("删除 Chroma 向量失败 document_id=%s", document_id)
                raise DocumentDeletionError("向量索引删除失败，请稍后重试。") from exc

            try:
                self._documents.delete_document(document_id)
            except Exception as exc:
                logger.exception("删除 SQLite 文档失败 document_id=%s", document_id)
                raise DocumentDeletionError("文档记录删除失败，请稍后重试。") from exc

            try:
                self._bm25.rebuild(self._documents.list_chunks())
            except Exception as exc:
                logger.exception("删除文档后重建 BM25 失败 document_id=%s", document_id)
                raise DocumentDeletionError("关键词索引重建失败，请稍后重试。") from exc

            try:
                source_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.exception("删除原始 PDF 失败 path=%s", source_path)
                raise DocumentDeletionError("原始 PDF 删除失败，请稍后重试。") from exc

            logger.info("文档已删除 document_id=%s", document_id)

    # 5
    def _clear_document_index(self, document_id: str) -> None:
        try:
            self._chroma.delete_document(document_id)
        except Exception:
            logger.exception("清理 Chroma 残留失败 document_id=%s", document_id)

        try:
            self._documents.delete_chunks(document_id)
        except Exception:
            logger.exception("清理 SQLite 分块失败 document_id=%s", document_id)

        try:
            self._bm25.rebuild(self._documents.list_chunks())
        except Exception:
            logger.exception("清理索引后重建 BM25 失败 document_id=%s", document_id)

    # 检查 SQLite 和 Chroma 实际保存的文本块数量是否与预期一致
    def _verify_index_consistency(self, document_id: str, expected_count: int) -> None:
        try:
            sqlite_count = self._documents.count_chunks(document_id)
        except Exception as exc:
            raise _IndexingStageError("分块数据保存失败，请重新提交索引。") from exc

        try:
            chroma_count = self._chroma.count_document(document_id)
        except Exception as exc:
            raise _IndexingStageError(
                "向量索引写入失败，请检查 Embedding 服务配置和网络连接。"
            ) from exc

        if sqlite_count != expected_count or chroma_count != expected_count:
            logger.error(
                "索引一致性校验失败 document_id=%s expected=%d sqlite=%d chroma=%d",
                document_id,
                expected_count,
                sqlite_count,
                chroma_count,
            )
            raise _IndexingStageError(
                "索引一致性校验失败，已清理不完整索引，请重新提交。"
            )


    @staticmethod
    def _pdf_error_message(exc: PdfExtractionError) -> str:
        safe_messages = {
            "PDF 已加密，暂不支持解析。",
            "PDF 不包含任何页面。",
            "PDF 未提取到文本，首版暂不支持扫描件或纯图片 PDF。",
        }
        message = str(exc)
        if message in safe_messages:
            return message
        return "文档解析失败，请确认 PDF 可以正常读取。"

