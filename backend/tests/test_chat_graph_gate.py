import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.domain.models import RefusalCode, RetrievedChunk
from app.generation.chat_graph import ChatGraphService
from app.generation.citation_service import CitationService
from app.generation.schemas import GenerationResult
from app.retrieval.context_selector import ContextSelector


class FakeRetriever:
    def __init__(self, relevant: list[RetrievedChunk]):
        self._relevant = relevant

    def filter_relevant(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return self._relevant


def candidate() -> RetrievedChunk:
    return RetrievedChunk(
        "doc-1:0",
        "doc-1",
        "manual.pdf",
        "石头",
        "G20",
        1,
        0,
        "尘盒清理说明",
        0.2,
        1.0,
        0.03,
        ("dense", "bm25"),
    )


def build_service(relevant: list[RetrievedChunk]) -> ChatGraphService:
    settings = Settings()
    service = object.__new__(ChatGraphService)
    service._settings = settings
    service._retriever = FakeRetriever(relevant)
    service._context_selector = ContextSelector(settings)
    service._citation_service = CitationService()
    return service


def test_gate_distinguishes_no_recall():
    service = build_service([])

    result = service._gate({"candidates": []})

    assert result["refusal_code"] == RefusalCode.NO_RECALL
    assert result["refused"] is True


def test_gate_distinguishes_low_relevance():
    weak = candidate()
    service = build_service([])

    result = service._gate({"candidates": [weak]})
    refusal = service._refuse(result)

    assert result["refusal_code"] == RefusalCode.LOW_RELEVANCE
    assert "相似度不足" in refusal["answer"]


def test_gate_builds_context_for_relevant_candidates():
    relevant = candidate()
    service = build_service([relevant])

    result = service._gate({"candidates": [relevant]})

    assert result["refused"] is False
    assert result["refusal_code"] is None
    assert result["contexts"][0].chunk_ids == ("doc-1:0",)


def test_valid_citation_promotes_raw_answer_to_final_answer():
    service = build_service([candidate()])
    gate_result = service._gate({"candidates": [candidate()]})
    source_result = service._prepare_sources(gate_result)

    result = service._validate_citations(
        {
            **source_result,
            "generation_result": GenerationResult(
                outcome="answer",
                answer="清理前请先关闭设备电源。[S1]",
            ),
        }
    )

    assert result["answer"] == "清理前请先关闭设备电源。[S1]"
    assert result["citation_valid"] is True
    assert result["cited_source_ids"] == ["S1"]
    assert result["refused"] is False


def test_invalid_citation_hides_raw_model_answer():
    service = build_service([candidate()])
    gate_result = service._gate({"candidates": [candidate()]})
    source_result = service._prepare_sources(gate_result)
    raw_answer = "这是不能暴露给用户的原始回答。[S99]"

    result = service._validate_citations(
        {
            **source_result,
            "generation_result": GenerationResult(
                outcome="answer",
                answer=raw_answer,
            ),
        }
    )

    assert raw_answer not in result["answer"]
    assert "不能暴露" not in result["answer"]
    assert result["citation_valid"] is False
    assert result["cited_source_ids"] == []
    assert result["refused"] is True
    assert result["refusal_code"] == RefusalCode.INVALID_CITATION


def test_insufficient_evidence_is_a_structured_business_refusal():
    service = build_service([candidate()])
    state = {
        "generation_result": GenerationResult(
            outcome="refuse",
            refusal_code="insufficient_evidence",
        )
    }

    assert service._route_generation(state) == "refuse"
    result = service._refuse_insufficient_evidence(state)

    assert result["refused"] is True
    assert result["refusal_code"] == RefusalCode.INSUFFICIENT_EVIDENCE
    assert result["cited_source_ids"] == []
    assert "没有足够证据" in result["answer"]


@pytest.mark.parametrize(
    "payload",
    [
        {"outcome": "answer", "answer": None, "refusal_code": None},
        {
            "outcome": "answer",
            "answer": "包含回答。[S1]",
            "refusal_code": "insufficient_evidence",
        },
        {"outcome": "refuse", "answer": "不应出现的正文", "refusal_code": "insufficient_evidence"},
        {"outcome": "refuse", "answer": None, "refusal_code": None},
    ],
)
def test_generation_result_rejects_conflicting_fields(payload: dict):
    with pytest.raises(ValidationError):
        GenerationResult.model_validate(payload)
