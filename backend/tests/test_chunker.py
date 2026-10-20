from app.config.settings import Settings
from app.domain.models import ExtractedPage
from app.ingestion.chunker import DocumentChunker


def test_recursive_chunking_preserves_page_numbers():
    settings = Settings(chunk_strategy="recursive", chunk_size=200, chunk_overlap=20)
    text = "第一段内容。" * 30
    chunks = DocumentChunker(settings).chunk(
        document_id="doc-1",
        document_name="manual.pdf",
        brand="石头",
        model="G20",
        pages=[ExtractedPage(page_number=3, text=text)],
    )

    assert chunks
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert all(chunk.model == "G20" for chunk in chunks)

