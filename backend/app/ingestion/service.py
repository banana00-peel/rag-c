from pathlib import Path

from app.config.settings import Settings
from app.domain.models import ExtractedPage
# from app.ingestion.catalog import detect_brand, detect_model
from app.ingestion.pdf import extract_pdf_pages

# 文档提取器
class PdfIngestionService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def inspect(self, path: str | Path) -> list[ExtractedPage]:
        pages = extract_pdf_pages(path)
        # preview = "\n".join(page.text for page in pages[:3])
        return pages

