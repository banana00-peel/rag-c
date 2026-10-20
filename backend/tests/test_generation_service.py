import pytest

from app.generation.schemas import GenerationResult
from app.generation.service import DeepSeekStructuredGenerator, GenerationOutputError


class FakeStructuredModel:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):
        return self._result


def build_generator(result) -> DeepSeekStructuredGenerator:
    generator = object.__new__(DeepSeekStructuredGenerator)
    generator._structured_llm = FakeStructuredModel(result)
    return generator


def test_structured_generator_returns_validated_result():
    expected = GenerationResult(
        outcome="refuse",
        refusal_code="insufficient_evidence",
    )
    generator = build_generator(expected)

    result = generator.generate("水箱容量是多少？", [])

    assert result is expected


def test_structured_generator_rejects_unexpected_result_type():
    generator = build_generator(
        {
            "outcome": "refuse",
            "refusal_code": "insufficient_evidence",
        }
    )

    with pytest.raises(GenerationOutputError, match="结构化结果"):
        generator.generate("水箱容量是多少？", [])
