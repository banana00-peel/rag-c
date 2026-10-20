from app.config.settings import Settings
from app.domain.models import RetrievedChunk
from app.retrieval.context_selector import ContextSelector


def candidate(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    page_number: int,
    text: str,
    rrf_score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        brand="石头",
        model="G20",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        dense_distance=0.2,
        bm25_score=1.0,
        rrf_score=rrf_score,
        matched_channels=("dense", "bm25"),
    )


def test_selector_limits_each_document_and_merges_adjacent_chunks():
    selector = ContextSelector(
        Settings(
            context_top_k=5,
            max_context_chunks_per_document=2,
            context_merge_max_chars=200,
        )
    )
    overlap = "共享内容" * 3
    candidates = [
        candidate("doc-1:0", "doc-1", 0, 1, f"清理前先断电，{overlap}", 0.5),
        candidate("doc-1:1", "doc-1", 1, 2, f"{overlap}再取出尘盒", 0.4),
        candidate("doc-1:2", "doc-1", 2, 2, "不应进入上下文", 0.3),
        candidate("doc-2:0", "doc-2", 0, 4, "另一份说明书", 0.2),
    ]

    contexts = selector.select(candidates)

    assert len(contexts) == 2
    assert contexts[0].chunk_ids == ("doc-1:0", "doc-1:1")
    assert contexts[0].page_start == 1
    assert contexts[0].page_end == 2
    assert contexts[0].text == f"清理前先断电，{overlap}再取出尘盒"
    assert contexts[0].dense_score == 0.8
    assert contexts[1].document_id == "doc-2"


def test_selector_deduplicates_normalized_content_within_document():
    selector = ContextSelector(Settings(context_top_k=5, max_context_chunks_per_document=3))
    candidates = [
        candidate("doc-1:0", "doc-1", 0, 1, "相同 内容", 0.5),
        candidate("doc-1:1", "doc-1", 1, 1, "相同内容", 0.4),
        candidate("doc-2:0", "doc-2", 0, 1, "相同内容", 0.3),
    ]

    contexts = selector.select(candidates)

    assert [context.document_id for context in contexts] == ["doc-1", "doc-2"]
    assert contexts[0].chunk_ids == ("doc-1:0",)


def test_selector_respects_merged_context_character_limit():
    selector = ContextSelector(
        Settings(context_top_k=5, max_context_chunks_per_document=3, context_merge_max_chars=200)
    )
    candidates = [
        candidate("doc-1:0", "doc-1", 0, 1, "甲" * 150, 0.5),
        candidate("doc-1:1", "doc-1", 1, 1, "乙" * 150, 0.4),
    ]

    contexts = selector.select(candidates)

    assert len(contexts) == 2


def test_selector_uses_full_context_budget_when_only_one_document_exists():
    selector = ContextSelector(
        Settings(
            context_top_k=5,
            max_context_chunks_per_document=2,
            context_merge_max_chars=200,
        )
    )
    candidates = [
        candidate(f"doc-1:{index}", "doc-1", index * 2, index + 1, f"片段 {index}", 1 - index / 10)
        for index in range(5)
    ]

    contexts = selector.select(candidates)

    assert len(contexts) == 5
