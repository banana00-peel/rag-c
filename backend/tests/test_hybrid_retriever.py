from pathlib import Path

import pytest

from app.config.settings import Settings
from app.domain.models import ChunkRecord, RetrievedChunk
from app.indexing.bm25_store import BM25Store
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_analyzer import QueryPlan


class FakeChroma:
    def __init__(self, results: dict[str, list[dict]]):
        self._results = results

    def query(self, query: str, **kwargs) -> list[dict]:
        return self._results.get(query, [])


class FakeBM25:
    def __init__(self, results: dict[str, list[dict]]):
        self._results = results
        self.min_scores: list[float] = []

    def search(self, query: str, **kwargs) -> list[dict]:
        self.min_scores.append(kwargs["min_score"])
        return self._results.get(query, [])


def dense_hit(chunk_id: str, distance: float, chunk_index: int = 0) -> dict:
    return {
        "id": chunk_id,
        "text": "尘盒清理说明",
        "distance": distance,
        "metadata": {
            "document_id": "doc-1",
            "document_name": "manual.pdf",
            "brand": "石头",
            "model": "G20",
            "page_number": 3,
            "chunk_index": chunk_index,
        },
    }


def keyword_hit(chunk: ChunkRecord, score: float) -> dict:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "chunk": chunk,
        "score": score,
    }


def test_hybrid_retriever_fuses_channels_and_keeps_best_scores():
    chunk = ChunkRecord(
        "doc-1:0", "doc-1", "manual.pdf", "石头", "G20", 3, 0, "尘盒清理说明"
    )
    chroma = FakeChroma(
        {
            "原始问题": [dense_hit(chunk.id, 0.4)],
            "扩展问题": [dense_hit(chunk.id, 0.2)],
        }
    )
    bm25 = FakeBM25({"原始问题": [keyword_hit(chunk, 2.5)]})
    settings = Settings(rrf_k=60, min_bm25_score=0.0)
    retriever = HybridRetriever(settings, chroma, bm25)
    plan = QueryPlan("原始问题", "原始问题", "G20", "通用咨询", ("原始问题", "扩展问题"))

    results = retriever.retrieve(plan)

    assert len(results) == 1
    assert results[0].dense_distance == 0.2
    assert results[0].dense_score == pytest.approx(0.8)
    assert results[0].bm25_score == 2.5
    assert results[0].rrf_score == pytest.approx(3 / 61)
    assert results[0].chunk_index == 0
    assert results[0].matched_channels == ("dense", "bm25")
    assert bm25.min_scores == [0.0, 0.0]


def test_hybrid_retriever_exposes_channel_trace_without_changing_fused_results():
    chunk = ChunkRecord(
        "doc-1:0", "doc-1", "manual.pdf", "石头", "G20", 3, 0, "尘盒清理说明"
    )
    chroma = FakeChroma({"问题": [dense_hit(chunk.id, 0.25)]})
    bm25 = FakeBM25({"问题": [keyword_hit(chunk, 2.5)]})
    retriever = HybridRetriever(Settings(rrf_k=60), chroma, bm25)
    plan = QueryPlan("问题", "问题", "G20", "通用咨询", ("问题",))

    trace = retriever.retrieve_with_trace(plan)

    assert [hit.id for hit in trace.dense_hits] == [chunk.id]
    assert trace.dense_hits[0].rank == 1
    assert trace.dense_hits[0].dense_distance == 0.25
    assert trace.dense_hits[0].query_variants == ("问题",)
    assert [hit.id for hit in trace.bm25_hits] == [chunk.id]
    assert trace.bm25_hits[0].bm25_score == 2.5
    assert list(trace.fused_hits) == retriever.retrieve(plan)


def test_hybrid_retriever_filters_relevance_per_channel():
    settings = Settings(max_dense_distance=0.75, min_bm25_score=0.0)
    retriever = HybridRetriever(settings, FakeChroma({}), FakeBM25({}))
    candidates = [
        RetrievedChunk(
            "weak", "doc-1", "a.pdf", "石头", "G20", 1, 0, "弱结果", 0.9, None, 0.3, ("dense",)
        ),
        RetrievedChunk(
            "dense", "doc-1", "a.pdf", "石头", "G20", 1, 1, "向量结果", 0.5, None, 0.2, ("dense",)
        ),
        RetrievedChunk(
            "bm25", "doc-2", "b.pdf", "石头", "G20", 2, 0, "关键词结果", 0.9, 1.2, 0.1, ("dense", "bm25")
        ),
    ]

    relevant = retriever.filter_relevant(candidates)

    assert [candidate.id for candidate in relevant] == ["dense", "bm25"]


def test_bm25_does_not_return_zero_score_chunks(tmp_path: Path):
    store = BM25Store(tmp_path / "bm25.json")
    store.rebuild(
        [
            ChunkRecord("doc-1:0", "doc-1", "a.pdf", "石头", "G20", 1, 0, "清理尘盒"),
            ChunkRecord("doc-2:0", "doc-2", "b.pdf", "石头", "S5", 1, 0, "连接网络"),
        ]
    )

    assert store.search("完全不存在的词", top_k=5, min_score=0.0) == []
