from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.config.settings import Settings
from app.domain.models import DocumentStatus, RetrievalChannelHit, RetrievedChunk
from app.evaluation.metrics import (
    ThresholdCase,
    average_metric_rows,
    calibrate_thresholds,
    is_explicit_knowledge_refusal,
    keyword_coverage,
    predict_retrieval_behavior,
    ranking_metrics,
    score_thresholds,
)
from app.evaluation.schemas import EvaluationCase, EvaluationDataError
from app.generation.chat_graph import ChatGraphService
from app.persistence.repository import DocumentRepository
from app.retrieval.context_selector import ContextSelector
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_analyzer import QueryAnalyzer


EvaluationMode = Literal["retrieval", "end-to-end"]


class EvaluationRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        documents: DocumentRepository,
        retriever: HybridRetriever,
        chat: ChatGraphService,
    ):
        self._settings = settings
        self._documents = documents
        self._retriever = retriever
        self._chat = chat
        self._analyzer = QueryAnalyzer()
        self._context_selector = ContextSelector(settings)

    def run(
        self,
        cases: list[EvaluationCase],
        *,
        dataset_path: str | Path,
        mode: EvaluationMode,
    ) -> dict:
        documents = self._validate_documents(cases)
        case_results: list[dict] = []
        threshold_inputs: list[tuple[EvaluationCase, tuple[RetrievedChunk, ...]]] = []

        for case in cases:
            plan = self._analyzer.analyze(case.question, case.model)
            trace = self._retriever.retrieve_with_trace(plan)
            fused_candidates = tuple(trace.fused_hits)
            threshold_inputs.append((case, fused_candidates))
            relevant = self._retriever.filter_relevant(list(fused_candidates))
            contexts = self._context_selector.select(relevant)
            predicted_behavior, refusal_code = predict_retrieval_behavior(
                fused_candidates,
                max_dense_distance=self._settings.max_dense_distance,
                min_bm25_score=self._settings.min_bm25_score,
            )

            result = {
                "id": case.id,
                "split": case.split,
                "question": case.question,
                "model": case.model,
                "category": case.category,
                "expected_behavior": case.expected_behavior,
                "retrieval_expectation": case.retrieval_expectation,
                "gold_pages": case.gold_pages,
                "predicted_behavior": predicted_behavior,
                "predicted_refusal_code": refusal_code,
                "retrieval": {
                    "dense": [self._serialize_channel_hit(hit) for hit in trace.dense_hits],
                    "bm25": [self._serialize_channel_hit(hit) for hit in trace.bm25_hits],
                    "fused": [
                        self._serialize_fused_hit(hit, rank)
                        for rank, hit in enumerate(fused_candidates, start=1)
                    ],
                    "selected_contexts": [
                        {
                            "chunk_ids": list(context.chunk_ids),
                            "document_id": context.document_id,
                            "page_start": context.page_start,
                            "page_end": context.page_end,
                            "dense_distance": context.dense_distance,
                            "bm25_score": context.bm25_score,
                            "rrf_score": context.rrf_score,
                        }
                        for context in contexts
                    ],
                },
                "metrics": self._case_retrieval_metrics(
                    case,
                    dense_pages=[{hit.page_number} for hit in trace.dense_hits],
                    bm25_pages=[{hit.page_number} for hit in trace.bm25_hits],
                    fused_pages=[{hit.page_number} for hit in fused_candidates],
                    context_pages=[
                        set(range(context.page_start, context.page_end + 1))
                        for context in contexts
                    ],
                ),
            }
            if mode == "end-to-end":
                result["generation"] = self._evaluate_generation(case)
            case_results.append(result)

        calibration = self._calibrate(threshold_inputs)
        report = {
            "metadata": self._build_metadata(dataset_path, mode, documents, cases),
            "summary": {
                "retrieval": self._summarize_retrieval(case_results),
                "refusal": self._summarize_refusal(case_results),
                "threshold_calibration": calibration,
            },
            "cases": case_results,
        }
        if mode == "end-to-end":
            report["summary"]["generation"] = self._summarize_generation(case_results)
        return report

    def _validate_documents(self, cases: list[EvaluationCase]) -> list:
        snapshots = []
        seen_hashes: set[str] = set()
        for case in cases:
            if not case.document_sha256 or case.document_sha256 in seen_hashes:
                continue
            document = self._documents.get_by_sha256(case.document_sha256)
            if document is None:
                raise EvaluationDataError(
                    f"样本 {case.id} 对应的文档尚未上传：{case.document_sha256}"
                )
            if document.status != DocumentStatus.COMPLETED:
                raise EvaluationDataError(
                    f"样本 {case.id} 对应的文档尚未完成索引：{document.filename}"
                )
            seen_hashes.add(case.document_sha256)
            snapshots.append(document)
        return snapshots

    def _calibrate(
        self,
        inputs: list[tuple[EvaluationCase, tuple[RetrievedChunk, ...]]],
    ) -> dict:
        calibration_cases = [
            ThresholdCase(case.retrieval_expectation, candidates)
            for case, candidates in inputs
            if case.split == "calibration"
            and case.retrieval_expectation != "retrieve_for_refusal"
        ]
        test_cases = [
            ThresholdCase(case.retrieval_expectation, candidates)
            for case, candidates in inputs
            if case.split == "test"
            and case.retrieval_expectation != "retrieve_for_refusal"
        ]
        calibration = calibrate_thresholds(calibration_cases)
        recommendation = calibration["recommendation"]
        calibration["current_thresholds"] = {
            "max_dense_distance": self._settings.max_dense_distance,
            "min_bm25_score": self._settings.min_bm25_score,
            **score_thresholds(
                calibration_cases,
                max_dense_distance=self._settings.max_dense_distance,
                min_bm25_score=self._settings.min_bm25_score,
            ),
        }
        current_test_metrics = score_thresholds(
            test_cases,
            max_dense_distance=self._settings.max_dense_distance,
            min_bm25_score=self._settings.min_bm25_score,
        )
        recommended_test_metrics = score_thresholds(
            test_cases,
            max_dense_distance=recommendation["max_dense_distance"],
            min_bm25_score=recommendation["min_bm25_score"],
        )
        calibration["test_comparison"] = {
            "current": current_test_metrics,
            "recommended": recommended_test_metrics,
        }
        holdout_passed = (
            recommended_test_metrics["balanced_accuracy"]
            > current_test_metrics["balanced_accuracy"]
            and recommended_test_metrics["answer_recall"]
            >= current_test_metrics["answer_recall"]
        )
        calibration["holdout_validation"] = {
            "passed": holdout_passed,
            "decision": (
                "候选阈值通过独立测试，可以进入人工复核。"
                if holdout_passed
                else "候选阈值未改善独立测试结果，保持当前线上配置。"
            ),
        }
        return calibration

    def _evaluate_generation(self, case: EvaluationCase) -> dict:
        state = self._chat.answer(case.question, case.model)
        refused = bool(state.get("refused", False))
        refusal_code = state.get("refusal_code")
        source_by_id = {
            source.source_id: source for source in state.get("citation_sources", [])
        }
        cited_sources = [
            source_by_id[source_id]
            for source_id in state.get("cited_source_ids", [])
            if source_id in source_by_id
        ]
        cited_page_sets = [
            set(range(source.context.page_start, source.context.page_end + 1))
            for source in cited_sources
        ]
        relevant_citations = sum(
            bool(set(case.gold_pages).intersection(pages)) for pages in cited_page_sets
        )
        citation_precision = (
            relevant_citations / len(cited_page_sets) if cited_page_sets else 0.0
        )
        citation_gold_hit = bool(
            set(case.gold_pages).intersection(set().union(*cited_page_sets))
        ) if cited_page_sets else False
        answer = state.get("answer", "")
        refusal_code_value = str(refusal_code) if refusal_code is not None else None
        allowed_codes = {str(code) for code in case.allowed_refusal_codes}
        structured_refusal_success = refused and refusal_code_value in allowed_codes
        semantic_refusal = is_explicit_knowledge_refusal(answer)
        if case.expected_behavior == "answer":
            success = not refused and citation_gold_hit
        else:
            success = structured_refusal_success
        return {
            "answer": answer,
            "refused": refused,
            "refusal_code": refusal_code_value,
            "citation_valid": bool(state.get("citation_valid", False)),
            "cited_source_ids": list(state.get("cited_source_ids", [])),
            "cited_pages": [sorted(pages) for pages in cited_page_sets],
            "citation_gold_hit": citation_gold_hit,
            "citation_precision": citation_precision,
            "keyword_coverage": keyword_coverage(answer, case.expected_keywords),
            "semantic_refusal": semantic_refusal,
            "structured_refusal_success": structured_refusal_success,
            "success": success,
        }

    def _build_metadata(
        self,
        dataset_path: str | Path,
        mode: EvaluationMode,
        documents: list,
        cases: list[EvaluationCase],
    ) -> dict:
        return {
            "dataset": Path(dataset_path).name,
            "mode": mode,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(cases),
            "calibration_count": sum(case.split == "calibration" for case in cases),
            "test_count": sum(case.split == "test" for case in cases),
            "documents": [
                {
                    "filename": document.filename,
                    "sha256": document.sha256,
                    "model": document.model,
                    "page_count": document.page_count,
                    "chunk_count": document.chunk_count,
                }
                for document in documents
            ],
            "configuration": {
                "embedding_model": self._settings.embedding_model,
                "generation_model": self._settings.deepseek_model,
                "chunk_strategy": self._settings.chunk_strategy,
                "chunk_size": self._settings.chunk_size,
                "chunk_overlap": self._settings.chunk_overlap,
                "dense_top_k": self._settings.dense_top_k,
                "bm25_top_k": self._settings.bm25_top_k,
                "context_top_k": self._settings.context_top_k,
                "rrf_k": self._settings.rrf_k,
                "max_dense_distance": self._settings.max_dense_distance,
                "min_bm25_score": self._settings.min_bm25_score,
            },
        }

    @staticmethod
    def _case_retrieval_metrics(
        case: EvaluationCase,
        *,
        dense_pages: list[set[int]],
        bm25_pages: list[set[int]],
        fused_pages: list[set[int]],
        context_pages: list[set[int]],
    ) -> dict:
        if case.expected_behavior != "answer":
            return {}
        return {
            "dense": ranking_metrics(dense_pages, case.gold_pages),
            "bm25": ranking_metrics(bm25_pages, case.gold_pages),
            "fused": ranking_metrics(fused_pages, case.gold_pages),
            "context": ranking_metrics(context_pages, case.gold_pages),
        }

    @staticmethod
    def _summarize_retrieval(results: list[dict]) -> dict:
        summary: dict[str, dict] = {}
        for split in ("all", "calibration", "test"):
            selected = [
                result
                for result in results
                if result["expected_behavior"] == "answer"
                and (split == "all" or result["split"] == split)
            ]
            summary[split] = {
                channel: average_metric_rows(
                    [result["metrics"][channel] for result in selected]
                )
                for channel in ("dense", "bm25", "fused", "context")
            }
        return summary

    @staticmethod
    def _summarize_refusal(results: list[dict]) -> dict:
        summary: dict[str, dict] = {}
        for split in ("all", "calibration", "test"):
            selected = [
                result for result in results if split == "all" or result["split"] == split
            ]
            answer_cases = [
                result for result in selected if result["retrieval_expectation"] == "answer"
            ]
            refusal_cases = [
                result for result in selected if result["retrieval_expectation"] == "refuse"
            ]
            generation_refusal_cases = [
                result
                for result in selected
                if result["retrieval_expectation"] == "retrieve_for_refusal"
            ]
            summary[split] = {
                "answer_recall": (
                    sum(result["predicted_behavior"] == "answer" for result in answer_cases)
                    / len(answer_cases)
                    if answer_cases
                    else 0.0
                ),
                "refusal_accuracy": (
                    sum(result["predicted_behavior"] == "refuse" for result in refusal_cases)
                    / len(refusal_cases)
                    if refusal_cases
                    else 0.0
                ),
                "generation_refusal_case_count": len(generation_refusal_cases),
            }
        return summary

    @staticmethod
    def _summarize_generation(results: list[dict]) -> dict:
        answer_cases = [
            result for result in results if result["expected_behavior"] == "answer"
        ]
        refusal_cases = [
            result for result in results if result["expected_behavior"] == "refuse"
        ]
        return {
            "answer_case_count": len(answer_cases),
            "refusal_case_count": len(refusal_cases),
            "citation_validation_rate": (
                sum(result["generation"]["citation_valid"] for result in answer_cases)
                / len(answer_cases)
                if answer_cases
                else 0.0
            ),
            "citation_gold_hit_rate": (
                sum(result["generation"]["citation_gold_hit"] for result in answer_cases)
                / len(answer_cases)
                if answer_cases
                else 0.0
            ),
            "average_citation_precision": (
                sum(result["generation"]["citation_precision"] for result in answer_cases)
                / len(answer_cases)
                if answer_cases
                else 0.0
            ),
            "average_keyword_coverage": (
                sum(result["generation"]["keyword_coverage"] for result in answer_cases)
                / len(answer_cases)
                if answer_cases
                else 0.0
            ),
            "refusal_accuracy": (
                sum(result["generation"]["success"] for result in refusal_cases)
                / len(refusal_cases)
                if refusal_cases
                else 0.0
            ),
            "structured_refusal_accuracy": (
                sum(
                    result["generation"]["structured_refusal_success"]
                    for result in refusal_cases
                )
                / len(refusal_cases)
                if refusal_cases
                else 0.0
            ),
            "semantic_refusal_rate": (
                sum(result["generation"]["semantic_refusal"] for result in refusal_cases)
                / len(refusal_cases)
                if refusal_cases
                else 0.0
            ),
            "unstructured_refusal_count": sum(
                result["generation"]["semantic_refusal"]
                and not result["generation"]["structured_refusal_success"]
                for result in refusal_cases
            ),
            "invalid_citation_count": sum(
                result["generation"]["refusal_code"] == "invalid_citation"
                for result in refusal_cases
            ),
            "end_to_end_success_rate": (
                sum(result["generation"]["success"] for result in results) / len(results)
                if results
                else 0.0
            ),
        }

    @staticmethod
    def _serialize_channel_hit(hit: RetrievalChannelHit) -> dict:
        return {
            "chunk_id": hit.id,
            "document_id": hit.document_id,
            "page_number": hit.page_number,
            "chunk_index": hit.chunk_index,
            "rank": hit.rank,
            "query_variants": list(hit.query_variants),
            "channel_rrf_score": hit.channel_rrf_score,
            "dense_distance": hit.dense_distance,
            "bm25_score": hit.bm25_score,
        }

    @staticmethod
    def _serialize_fused_hit(hit: RetrievedChunk, rank: int) -> dict:
        return {
            "chunk_id": hit.id,
            "document_id": hit.document_id,
            "page_number": hit.page_number,
            "chunk_index": hit.chunk_index,
            "rank": rank,
            "dense_distance": hit.dense_distance,
            "bm25_score": hit.bm25_score,
            "rrf_score": hit.rrf_score,
            "matched_channels": list(hit.matched_channels),
        }
