import hashlib
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config.settings import Settings
from app.domain.models import DocumentSnapshot, DocumentStatus
from app.ingestion.catalog import detect_brand, detect_model, normalize_model
from app.ingestion.pdf import part_pdf_cut
from app.persistence.repository import DocumentRepository


class DocumentUploadError(ValueError):
    pass


class DocumentService:
    def __init__(self, settings: Settings, documents: DocumentRepository):
        self._settings = settings
        self._documents = documents

    async def upload(self, file: UploadFile) -> tuple[DocumentSnapshot, bool]:
        filename = self._safe_filename(file.filename or "uploaded.pdf")
        if not filename.lower().endswith(".pdf"):
            raise DocumentUploadError("只支持 PDF 文件。")

        document_id = str(uuid4())
        target_path = self._settings.upload_dir / f"{document_id}.pdf" # 用户上传的文档放在这个位置
        digest = hashlib.sha256() # 创建 SHA-256 计算器
        total_bytes = 0
        try:
            with target_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024): # 一边写，一边判断是否为空
                    total_bytes += len(chunk)
                    if total_bytes > self._settings.max_upload_bytes:
                        raise DocumentUploadError(
                            f"文件超过 {self._settings.max_upload_mb} MB 限制。"
                        )
                    digest.update(chunk)
                    output.write(chunk)
        except Exception:
            target_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        sha256 = digest.hexdigest()
        duplicate = self._documents.get_by_sha256(sha256)
        if duplicate is not None:
            target_path.unlink(missing_ok=True)
            return duplicate, True # True：表示文档重复

        try:
            pages = part_pdf_cut(target_path)
            preview = "\n".join(page.text for page in pages[:3])
            brand = detect_brand(filename, preview)
            model = normalize_model(detect_model(filename, preview))
        except Exception as exc:
            target_path.unlink(missing_ok=True)
            raise DocumentUploadError(str(exc)) from exc

        return (
            self._documents.create(
                document_id=document_id,
                filename=filename,
                stored_path=str(target_path),
                sha256=sha256,
                brand=brand,
                model=model,
            ),# 文档元数据入库
            False,
        )

    # 元数据更新
    def update_metadata(
        self,
        document_id: str,
        *,
        brand: str,
        model: str,
    ) -> DocumentSnapshot:
        document = self._documents.get_by_id(document_id)
        if document is None:
            raise KeyError(document_id)
        if document.status not in {DocumentStatus.PENDING, DocumentStatus.FAILED}:
            raise DocumentUploadError("只有待索引或索引失败的文档可以修改元数据。")
        normalized_model = normalize_model(model)
        if not brand.strip() or not normalized_model:
            raise DocumentUploadError("品牌和型号不能为空。")
        return self._documents.update_metadata(
            document_id,
            brand=brand,
            model=normalized_model,
        )

    @staticmethod
    def _safe_filename(filename: str) -> str:
        filename = Path(filename).name
        filename = re.sub(r"[^\w\u4e00-\u9fff.\- ()]", "_", filename)
        return filename[:255] or "uploaded.pdf"

