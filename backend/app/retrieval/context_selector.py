import re
from collections import defaultdict

from app.config.settings import Settings
from app.domain.models import ContextBlock, RetrievedChunk


class ContextSelector:
    """将排序候选整理为去重、多样且适合送入大模型的上下文。"""

    _MIN_OVERLAP_CHARS = 12

    def __init__(self, settings: Settings):
        self._settings = settings

    def select(self, candidates: list[RetrievedChunk]) -> list[ContextBlock]:
        selected = self._select_diverse_chunks(candidates)
        if not selected:
            return []

        original_ranks = {candidate.id: rank for rank, candidate in enumerate(selected)}
        chunks_by_document: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
        for candidate in selected:
            chunks_by_document[candidate.document_id].append(candidate)

        ranked_blocks: list[tuple[int, ContextBlock]] = []
        for document_chunks in chunks_by_document.values():
            document_chunks.sort(key=lambda item: item.chunk_index)
            for group in self._group_adjacent(document_chunks):
                best_rank = min(original_ranks[item.id] for item in group)
                ranked_blocks.append((best_rank, self._build_block(group)))

        ranked_blocks.sort(key=lambda item: item[0])
        return [block for _, block in ranked_blocks]

    def _select_diverse_chunks(
        self,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        document_counts: defaultdict[str, int] = defaultdict(int)
        seen_content: set[tuple[str, str]] = set()
        document_limit = (
            self._settings.context_top_k
            if len({candidate.document_id for candidate in candidates}) <= 1
            else self._settings.max_context_chunks_per_document
        )

        for candidate in candidates:
            content_key = (candidate.document_id, self._content_fingerprint(candidate.text))
            if content_key in seen_content:
                continue
            if (
                document_counts[candidate.document_id]
                >= document_limit
            ):
                continue

            selected.append(candidate)
            seen_content.add(content_key)
            document_counts[candidate.document_id] += 1
            if len(selected) >= self._settings.context_top_k:
                break

        return selected

    def _group_adjacent(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[list[RetrievedChunk]]:
        groups: list[list[RetrievedChunk]] = []
        for chunk in chunks:
            if not groups:
                groups.append([chunk])
                continue

            current = groups[-1]
            previous = current[-1]
            merged_text = self._merge_group_text([*current, chunk])
            is_adjacent = (
                chunk.chunk_index == previous.chunk_index + 1
                and 0 <= chunk.page_number - previous.page_number <= 1
            )
            if is_adjacent and len(merged_text) <= self._settings.context_merge_max_chars:
                current.append(chunk)
            else:
                groups.append([chunk])
        return groups

    def _build_block(self, chunks: list[RetrievedChunk]) -> ContextBlock:
        first = chunks[0]
        dense_distances = [
            chunk.dense_distance for chunk in chunks if chunk.dense_distance is not None
        ]
        bm25_scores = [chunk.bm25_score for chunk in chunks if chunk.bm25_score is not None]
        channels = {
            channel for chunk in chunks for channel in chunk.matched_channels
        }
        return ContextBlock(
            chunk_ids=tuple(chunk.id for chunk in chunks),
            document_id=first.document_id,
            document_name=first.document_name,
            brand=first.brand,
            model=first.model,
            page_start=min(chunk.page_number for chunk in chunks),
            page_end=max(chunk.page_number for chunk in chunks),
            text=self._merge_group_text(chunks),
            dense_distance=min(dense_distances) if dense_distances else None,
            bm25_score=max(bm25_scores) if bm25_scores else None,
            rrf_score=max(chunk.rrf_score for chunk in chunks),
            matched_channels=tuple(
                channel for channel in ("dense", "bm25") if channel in channels
            ),
        )

    @staticmethod
    def _content_fingerprint(text: str) -> str:
        return re.sub(r"\s+", "", text).casefold()

    @classmethod
    def _merge_group_text(cls, chunks: list[RetrievedChunk]) -> str:
        merged = chunks[0].text.strip()
        for chunk in chunks[1:]:
            merged = cls._merge_overlapping_text(merged, chunk.text.strip())
        return merged

    @staticmethod
    def _merge_overlapping_text(left: str, right: str) -> str:
        max_overlap = min(len(left), len(right))
        for size in range(max_overlap, ContextSelector._MIN_OVERLAP_CHARS - 1, -1):
            if left[-size:] == right[:size]:
                return left + right[size:]
        return f"{left}\n{right}"
