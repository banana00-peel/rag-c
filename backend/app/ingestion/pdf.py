from pathlib import Path

import pymupdf

from app.domain.models import ExtractedPage
from app.ingestion.catalog import normalize_text


class PdfExtractionError(ValueError):
    pass


def extract_pdf_pages(path: str | Path) -> list[ExtractedPage]:
    try:
        document = pymupdf.open(str(path))
    except Exception as exc:
        raise PdfExtractionError(f"无法打开 PDF: {exc}") from exc

    try:
        if document.is_encrypted:
            raise PdfExtractionError("PDF 已加密，暂不支持解析。")
        if document.page_count == 0:
            raise PdfExtractionError("PDF 不包含任何页面。")

        pages: list[ExtractedPage] = []
        for index in range(document.page_count):
            text = normalize_text(document.load_page(index).get_text("text")) # 进行文本清洗
            if text:
                pages.append(ExtractedPage(page_number=index + 1, text=text))

        if not pages:
            raise PdfExtractionError("PDF 未提取到文本，首版暂不支持扫描件或纯图片 PDF。")
        return pages
    finally:
        document.close()


def part_pdf_cut(path: str | Path, part_age: int = 3) -> list[ExtractedPage]:
    try:
        document = pymupdf.open(str(path))
    except Exception as exc:
        raise PdfExtractionError(f"无法打开 PDF: {exc}") from exc

    try:
        if document.is_encrypted:
            raise PdfExtractionError("PDF 已加密，暂不支持解析。")
        if document.page_count == 0:
            raise PdfExtractionError("PDF 不包含任何页面。")

        pages: list[ExtractedPage] = []
        preview_ages = min(part_age,document.page_count)
        for index in range(preview_ages):
            text = normalize_text(document.load_page(index).get_text("text")) # 进行文本清洗
            if text:
                pages.append(ExtractedPage(page_number=index + 1, text=text))

        if not pages:
            raise PdfExtractionError("PDF 未提取到文本，首版暂不支持扫描件或纯图片 PDF。")
        return pages
    finally:
        document.close()