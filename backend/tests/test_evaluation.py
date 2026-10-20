import json
from pathlib import Path

import pytest

from app.domain.models import RetrievedChunk
from app.evaluation.metrics import (
    ThresholdCase,
    calibrate_thresholds,
    is_explicit_knowledge_refusal,
    keyword_coverage,
    ranking_metrics,
)
from app.evaluation.report import write_report
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationDataError, load_dataset


DOCUMENT_SHA256 = "a" * 64


def answer_case(case_id: str, split: str) -> dict:
    return {
        "id": case_id,
        "split": split,
        "question": "如何清理尘盒？",
        "model": "G20",
        "category": "清洁与维护",
        "expected_behavior": "answer",
        "document_sha256": DOCUMENT_SHA256,
        "gold_pages": [11],
        "gold_evidence": ["使用清水清洗"],
        "expected_keywords": ["清水"],
    }


def refusal_case(case_id: str, split: str) -> dict:
    return {
        "id": case_id,
        "split": split,
        "question": "是否支持 HomeKit？",
        "model": "G20",
        "category": "知识库外",
        "expected_behavior": "refuse",
        "allowed_refusal_codes": ["no_recall", "low_relevance"],
    }


def retrieved_chunk(chunk_id: str, dense_distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id="doc-1",
        document_name="manual.pdf",
        brand="石头",
        model="G20",
        page_number=11,
        chunk_index=0,
        text="清理说明",
        dense_distance=dense_distance,
        bm25_score=None,
        rrf_score=0.03,
        matched_channels=("dense",),
    )


def test_load_dataset_validates_splits_and_ground_truth(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(answer_case("cal-answer", "calibration"), ensure_ascii=False),
                json.dumps(refusal_case("test-refuse", "test"), ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_dataset(dataset)

    assert [case.id for case in cases] == ["cal-answer", "test-refuse"]
    assert cases[0].gold_pages == [11]
    assert cases[0].retrieval_expectation == "answer"
    assert [str(code) for code in cases[1].allowed_refusal_codes] == [
        "no_recall",
        "low_relevance",
    ]


def test_refusal_case_can_require_retrieval_before_generation_refusal(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    calibration = refusal_case("cal-topic-only", "calibration")
    calibration["retrieval_expectation"] = "retrieve_for_refusal"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(calibration, ensure_ascii=False),
                json.dumps(answer_case("test-answer", "test"), ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_dataset(dataset)

    assert cases[0].expected_behavior == "refuse"
    assert cases[0].retrieval_expectation == "retrieve_for_refusal"


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    payload = json.dumps(answer_case("same-id", "calibration"), ensure_ascii=False)
    dataset.write_text(f"{payload}\n{payload}\n", encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="ID 重复"):
        load_dataset(dataset)


def test_ranking_metrics_use_page_level_ground_truth():
    metrics = ranking_metrics([{2}, {10, 11}, {16}], [11])

    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr_at_5"] == 0.5
    assert metrics["page_recall_at_5"] == 1.0


def test_threshold_calibration_preserves_answer_recall_and_rejects_weak_result():
    cases = [
        ThresholdCase("answer", (retrieved_chunk("answer", 0.6),)),
        ThresholdCase("refuse", (retrieved_chunk("negative", 0.8),)),
    ]

    result = calibrate_thresholds(cases)

    recommendation = result["recommendation"]
    assert recommendation["max_dense_distance"] == 0.6
    assert recommendation["answer_recall"] == 1.0
    assert recommendation["refusal_accuracy"] == 1.0


def test_keyword_coverage_ignores_spacing_and_case():
    coverage = keyword_coverage("仅支持 2.4 GHz，不支持 5GHZ。", ["2.4GHz", "5ghz"])

    assert coverage == 1.0


def test_explicit_refusal_does_not_treat_citation_failure_as_business_refusal():
    assert is_explicit_knowledge_refusal("根据当前知识库无法确认，建议联系人工客服。")
    assert not is_explicit_knowledge_refusal("生成的回答未通过来源引用校验。")


def test_dataset_accepts_insufficient_evidence_refusal_code(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    refusal = refusal_case("cal-insufficient", "calibration")
    refusal["retrieval_expectation"] = "retrieve_for_refusal"
    refusal["allowed_refusal_codes"] = ["insufficient_evidence"]
    dataset.write_text(
        "\n".join(
            [
                json.dumps(refusal, ensure_ascii=False),
                json.dumps(answer_case("test-answer", "test"), ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_dataset(dataset)

    assert str(cases[0].allowed_refusal_codes[0]) == "insufficient_evidence"


def test_semantic_refusal_cannot_replace_structured_refusal():
    class FakeChat:
        @staticmethod
        def answer(question: str, model: str | None):
            return {
                "answer": "根据当前知识库无法确认，建议联系人工客服。",
                "refused": False,
                "refusal_code": None,
                "citation_sources": [],
                "cited_source_ids": [],
            }

    case_payload = refusal_case("test-semantic-only", "test")
    case_payload["allowed_refusal_codes"] = ["insufficient_evidence"]
    case = EvaluationCase.model_validate(case_payload)
    runner = object.__new__(EvaluationRunner)
    runner._chat = FakeChat()

    result = runner._evaluate_generation(case)

    assert result["semantic_refusal"] is True
    assert result["structured_refusal_success"] is False
    assert result["success"] is False


def test_report_refuses_to_write_api_key_fields(tmp_path: Path):
    with pytest.raises(ValueError, match="API Key"):
        write_report({"api_key": "secret"}, tmp_path, "unsafe")
