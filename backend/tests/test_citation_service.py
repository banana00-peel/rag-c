import pytest

from app.api.routes import build_cited_source_responses
from app.domain.models import CitationSource, ContextBlock
from app.generation.citation_service import CitationService


def context(
    document_id: str = "doc-1",
    text: str = "清理前先关闭设备电源。",
) -> ContextBlock:
    return ContextBlock(
        chunk_ids=(f"{document_id}:0",),
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        brand="石头",
        model="G20",
        page_start=1,
        page_end=1,
        text=text,
        dense_distance=0.2,
        bm25_score=1.2,
        rrf_score=0.03,
        matched_channels=("dense", "bm25"),
    )


def test_assign_sources_uses_stable_request_scoped_ids():
    sources = CitationService.assign_sources([context("doc-1"), context("doc-2")])

    assert [source.source_id for source in sources] == ["S1", "S2"]
    assert [source.context.document_id for source in sources] == ["doc-1", "doc-2"]


def test_validate_accepts_known_sources_and_preserves_first_use_order():
    sources = CitationService.assign_sources([context("doc-1"), context("doc-2")])

    result = CitationService.validate("第二条依据 [S2]，第一条依据 [S1]，再次引用 [S2]。", sources)

    assert result.is_valid is True
    assert result.cited_source_ids == ("S2", "S1")
    assert result.invalid_references == ()


@pytest.mark.parametrize(
    "answer",
    [
        "引用不存在的来源 [S99]。",
        "引用模型生成的页码 [说明书.pdf，第99页]。",
        "使用格式错误的来源 [S01]。",
        "来源 ID 两侧包含空格 [ S1 ]。",
        "先给出合法引用 [S1]，再给出未闭合引用 [S2。",
    ],
)
def test_validate_rejects_unknown_or_untrusted_bracket_references(answer: str):
    sources = CitationService.assign_sources([context()])

    result = CitationService.validate(answer, sources)

    assert result.is_valid is False
    assert result.cited_source_ids == ()
    assert result.invalid_references


def test_validate_rejects_answer_without_citation():
    sources = CitationService.assign_sources([context()])

    result = CitationService.validate("清理前先关闭设备电源。", sources)

    assert result.is_valid is False
    assert result.invalid_references == ()
    assert "没有提供" in result.error_message


def test_api_returns_only_cited_sources_and_keeps_full_snippet():
    long_text = "原文" * 180
    sources = [
        CitationSource("S1", context("doc-1")),
        CitationSource("S2", context("doc-2", long_text)),
    ]

    responses = build_cited_source_responses(
        {
            "citation_sources": sources,
            "cited_source_ids": ["S2"],
        }
    )

    assert len(responses) == 1
    assert responses[0].source_id == "S2"
    assert responses[0].document_id == "doc-2"
    assert responses[0].snippet == long_text


def test_api_rejects_citation_without_trusted_source_mapping():
    with pytest.raises(ValueError, match="S2"):
        build_cited_source_responses(
            {
                "citation_sources": [CitationSource("S1", context())],
                "cited_source_ids": ["S2"],
            }
        )


def test_api_rejects_generated_answer_without_verified_sources():
    with pytest.raises(ValueError, match="没有已验证"):
        build_cited_source_responses(
            {
                "citation_sources": [],
                "cited_source_ids": [],
                "refused": False,
            }
        )
