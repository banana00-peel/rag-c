import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.domain.models import RetrievedChunk


_DENSE_THRESHOLDS = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
_BM25_THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class ThresholdCase:
    expected_behavior: str
    candidates: tuple[RetrievedChunk, ...]


def ranking_metrics(
    ranked_page_sets: Sequence[set[int] | frozenset[int]],
    gold_pages: Iterable[int],
) -> dict[str, float]:
    gold = set(gold_pages)
    first_relevant_rank: int | None = None
    for rank, pages in enumerate(ranked_page_sets, start=1):
        if gold.intersection(pages):
            first_relevant_rank = rank
            break

    result: dict[str, float] = {}
    for top_k in (1, 3, 5):
        retrieved = set().union(*ranked_page_sets[:top_k]) if ranked_page_sets else set()
        result[f"hit_at_{top_k}"] = float(bool(gold.intersection(retrieved)))
        result[f"page_recall_at_{top_k}"] = (
            len(gold.intersection(retrieved)) / len(gold) if gold else 0.0
        )
    result["mrr_at_5"] = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None and first_relevant_rank <= 5
        else 0.0
    )
    return result


def average_metric_rows(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def predict_retrieval_behavior(
    candidates: Sequence[RetrievedChunk],
    *,
    max_dense_distance: float,
    min_bm25_score: float,
) -> tuple[str, str | None]:
    if not candidates:
        return "refuse", "no_recall"
    for candidate in candidates:
        dense_relevant = (
            candidate.dense_distance is not None
            and candidate.dense_distance <= max_dense_distance
        )
        bm25_relevant = (
            candidate.bm25_score is not None
            and candidate.bm25_score > min_bm25_score
        )
        if dense_relevant or bm25_relevant:
            return "answer", None
    return "refuse", "low_relevance"


def score_thresholds(
    cases: Sequence[ThresholdCase],
    *,
    max_dense_distance: float,
    min_bm25_score: float,
) -> dict[str, float]:
    answer_cases = [case for case in cases if case.expected_behavior == "answer"]
    refusal_cases = [case for case in cases if case.expected_behavior == "refuse"]
    answered_correctly = sum(
        predict_retrieval_behavior(
            case.candidates,
            max_dense_distance=max_dense_distance,
            min_bm25_score=min_bm25_score,
        )[0]
        == "answer"
        for case in answer_cases
    )
    refused_correctly = sum(
        predict_retrieval_behavior(
            case.candidates,
            max_dense_distance=max_dense_distance,
            min_bm25_score=min_bm25_score,
        )[0]
        == "refuse"
        for case in refusal_cases
    )
    answer_recall = answered_correctly / len(answer_cases) if answer_cases else 0.0
    refusal_accuracy = refused_correctly / len(refusal_cases) if refusal_cases else 0.0
    return {
        "answer_recall": answer_recall,
        "refusal_accuracy": refusal_accuracy,
        "balanced_accuracy": (answer_recall + refusal_accuracy) / 2,
    }


def calibrate_thresholds(
    cases: Sequence[ThresholdCase],
    *,
    minimum_answer_recall: float = 0.9,
) -> dict:
    scored: list[dict] = []
    for dense_threshold in _DENSE_THRESHOLDS:
        for bm25_threshold in _BM25_THRESHOLDS:
            metrics = score_thresholds(
                cases,
                max_dense_distance=dense_threshold,
                min_bm25_score=bm25_threshold,
            )
            scored.append(
                {
                    "max_dense_distance": dense_threshold,
                    "min_bm25_score": bm25_threshold,
                    **metrics,
                    "meets_answer_recall_constraint": (
                        metrics["answer_recall"] >= minimum_answer_recall
                    ),
                }
            )

    feasible = [row for row in scored if row["meets_answer_recall_constraint"]]
    pool = feasible or scored
    recommendation = max(
        pool,
        key=lambda row: (
            row["refusal_accuracy"] if feasible else row["balanced_accuracy"],
            row["balanced_accuracy"],
            row["answer_recall"],
            -row["max_dense_distance"],
            row["min_bm25_score"],
        ),
    )
    top_candidates = sorted(
        pool,
        key=lambda row: (
            -row["balanced_accuracy"],
            -row["refusal_accuracy"],
            -row["answer_recall"],
            row["max_dense_distance"],
            -row["min_bm25_score"],
        ),
    )[:5]
    return {
        "minimum_answer_recall": minimum_answer_recall,
        "used_fallback_without_constraint": not bool(feasible),
        "recommendation": recommendation,
        "top_candidates": top_candidates,
    }


def keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> float:
    if not expected_keywords:
        return 1.0
    normalized_answer = _normalize_text(answer)
    matched = sum(
        _normalize_text(keyword) in normalized_answer for keyword in expected_keywords
    )
    return matched / len(expected_keywords)


def is_explicit_knowledge_refusal(answer: str) -> bool:
    """识别提示词约定的证据不足拒答，不把引用校验故障当作业务拒答。"""
    return "根据当前知识库无法确认" in answer


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()
