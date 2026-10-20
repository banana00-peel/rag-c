from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import Settings
from app.domain.models import ChunkRecord, ExtractedPage


@dataclass(frozen=True)
class ChunkingOptions:
    strategy: str
    chunk_size: int
    chunk_overlap: int


class DocumentChunker:
    def __init__(self, settings: Settings):
        self._options = ChunkingOptions(
            strategy=settings.chunk_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def chunk(
        self,
        *,
        document_id: str,
        document_name: str,
        brand: str,
        model: str,
        pages: list[ExtractedPage],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        chunk_index = 0
        for page in pages:
            pieces = self._split_page(page.text) # 对每页文档切割
            for text in pieces:
                normalized = text.strip()
                if not normalized:
                    continue
                chunks.append(
                    ChunkRecord(
                        id=f"{document_id}:{chunk_index}",
                        document_id=document_id,
                        document_name=document_name,
                        brand=brand,
                        model=model,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=normalized,
                    )
                )
                chunk_index += 1
        return chunks
    # TODO 性能优化
    #  _split_page 在循环里每次新建 splitter，对每页都重复构造一次。
    #  但如果文档页数很多，把 splitter 提到 __init__ 里复用会更快。

    # 文档切分选择：递归或自定义，chunk方法调用这个函数
    def _split_page(self, text: str) -> list[str]:
        if self._options.strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._options.chunk_size,
                chunk_overlap=self._options.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                length_function=len,
            )
            return splitter.split_text(text)
        return self._split_fixed(text)

    def _split_fixed(self, text: str) -> list[str]:
        size = self._options.chunk_size
        overlap = self._options.chunk_overlap if self._options.strategy == "overlap" else 0
        if len(text) <= size:
            return [text]

        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            pieces.append(text[start:end])
            if end >= len(text):
                break
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start
        return pieces

