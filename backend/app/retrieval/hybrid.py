from collections import defaultdict

from app.config.settings import Settings
from app.domain.models import RetrievalChannelHit, RetrievalTrace, RetrievedChunk
from app.indexing.bm25_store import BM25Store
from app.indexing.chroma_store import ChromaStore
from app.retrieval.query_analyzer import QueryPlan


class HybridRetriever:
    def __init__(self, settings: Settings, chroma: ChromaStore, bm25: BM25Store):
        self._settings = settings
        self._chroma = chroma
        self._bm25 = bm25

    def retrieve(self, plan: QueryPlan) -> list[RetrievedChunk]:
        return list(self.retrieve_with_trace(plan).fused_hits)

    def retrieve_with_trace(self, plan: QueryPlan) -> RetrievalTrace:
        """执行一次真实检索，并保留各通道排名供离线评测使用。"""
        dense_by_id: dict[str, dict] = {}
        bm25_by_id: dict[str, dict] = {}
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        dense_rrf_scores: defaultdict[str, float] = defaultdict(float)
        bm25_rrf_scores: defaultdict[str, float] = defaultdict(float)
        dense_variants: defaultdict[str, list[str]] = defaultdict(list)
        bm25_variants: defaultdict[str, list[str]] = defaultdict(list)

        for query in plan.variants:
            dense_results = self._chroma.query(
                query,
                top_k=self._settings.dense_top_k,
                model=plan.model,
            )
            for rank, item in enumerate(dense_results, start=1):
                chunk_id = item["id"]
                existing = dense_by_id.get(chunk_id)
                if existing is None or self._is_better_dense_hit(item, existing):
                    dense_by_id[chunk_id] = item
                contribution = 1 / (self._settings.rrf_k + rank)
                rrf_scores[chunk_id] += contribution
                dense_rrf_scores[chunk_id] += contribution
                if query not in dense_variants[chunk_id]:
                    dense_variants[chunk_id].append(query)

            bm25_results = self._bm25.search(
                query,
                top_k=self._settings.bm25_top_k,
                model=plan.model,
                min_score=self._settings.min_bm25_score,
            )
            for rank, item in enumerate(bm25_results, start=1):
                chunk_id = item["id"]
                existing = bm25_by_id.get(chunk_id)
                if existing is None or item["score"] > existing["score"]:
                    bm25_by_id[chunk_id] = item
                contribution = 1 / (self._settings.rrf_k + rank)
                rrf_scores[chunk_id] += contribution
                bm25_rrf_scores[chunk_id] += contribution
                if query not in bm25_variants[chunk_id]:
                    bm25_variants[chunk_id].append(query)

        candidates: list[RetrievedChunk] = []
        all_ids = set(dense_by_id) | set(bm25_by_id)
        for chunk_id in all_ids:
            dense = dense_by_id.get(chunk_id)
            keyword = bm25_by_id.get(chunk_id)
            source = dense or keyword
            if source is None:
                continue
            if dense is not None:
                metadata = dense["metadata"]
                dense_distance = dense["distance"]
                text = dense["text"]
            else:
                chunk = keyword["chunk"]
                metadata = {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "brand": chunk.brand,
                    "model": chunk.model,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }
                dense_distance = None
                text = keyword["text"]
            candidates.append(
                RetrievedChunk(
                    id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    document_name=str(metadata["document_name"]),
                    brand=str(metadata["brand"]),
                    model=str(metadata["model"]),
                    page_number=int(metadata["page_number"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=text,
                    dense_distance=dense_distance,
                    bm25_score=keyword["score"] if keyword else None,
                    rrf_score=rrf_scores[chunk_id],
                    matched_channels=tuple(
                        channel
                        for channel, matched in (
                            ("dense", dense is not None),
                            ("bm25", keyword is not None),
                        )
                        if matched
                    ),
                )
            )

        candidates.sort(key=lambda item: (-item.rrf_score, item.id))
        return RetrievalTrace(
            dense_hits=self._build_dense_trace(
                dense_by_id,
                dense_rrf_scores,
                dense_variants,
            ),
            bm25_hits=self._build_bm25_trace(
                bm25_by_id,
                bm25_rrf_scores,
                bm25_variants,
            ),
            fused_hits=tuple(candidates),
        )

    def filter_relevant(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """按各检索通道自己的量纲过滤，RRF 只承担排序职责。"""
        return [candidate for candidate in candidates if self._is_relevant(candidate)]

    def _is_relevant(self, candidate: RetrievedChunk) -> bool:
        dense_relevant = (
            candidate.dense_distance is not None
            and candidate.dense_distance <= self._settings.max_dense_distance
        )
        bm25_relevant = (
            candidate.bm25_score is not None
            and candidate.bm25_score > self._settings.min_bm25_score
        )
        return dense_relevant or bm25_relevant

    @staticmethod
    def _is_better_dense_hit(candidate: dict, existing: dict) -> bool:
        candidate_distance = candidate.get("distance")
        existing_distance = existing.get("distance")
        if candidate_distance is None:
            return False
        return existing_distance is None or candidate_distance < existing_distance

    @staticmethod
    def _build_dense_trace(
        hits_by_id: dict[str, dict],
        channel_scores: dict[str, float],
        variants: dict[str, list[str]],
    ) -> tuple[RetrievalChannelHit, ...]:
        ordered_ids = sorted(hits_by_id, key=lambda item_id: (-channel_scores[item_id], item_id))
        hits: list[RetrievalChannelHit] = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            item = hits_by_id[chunk_id]
            metadata = item["metadata"]
            hits.append(
                RetrievalChannelHit(
                    id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    document_name=str(metadata["document_name"]),
                    model=str(metadata["model"]),
                    page_number=int(metadata["page_number"]),
                    chunk_index=int(metadata["chunk_index"]),
                    rank=rank,
                    query_variants=tuple(variants[chunk_id]),
                    channel_rrf_score=channel_scores[chunk_id],
                    dense_distance=float(item["distance"]),
                )
            )
        return tuple(hits)

    @staticmethod
    def _build_bm25_trace(
        hits_by_id: dict[str, dict],
        channel_scores: dict[str, float],
        variants: dict[str, list[str]],
    ) -> tuple[RetrievalChannelHit, ...]:
        ordered_ids = sorted(hits_by_id, key=lambda item_id: (-channel_scores[item_id], item_id))
        hits: list[RetrievalChannelHit] = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            item = hits_by_id[chunk_id]
            chunk = item["chunk"]
            hits.append(
                RetrievalChannelHit(
                    id=chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    model=chunk.model,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    rank=rank,
                    query_variants=tuple(variants[chunk_id]),
                    channel_rrf_score=channel_scores[chunk_id],
                    bm25_score=float(item["score"]),
                )
            )
        return tuple(hits)

