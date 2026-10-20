import re
from dataclasses import dataclass

from app.domain.models import CitationSource, ContextBlock


_BRACKET_REFERENCE_PATTERN = re.compile(r"\[([^\[\]]*)\]")
_SOURCE_ID_PATTERN = re.compile(r"S[1-9]\d*")


@dataclass(frozen=True)
class CitationValidationResult:
    is_valid: bool
    cited_source_ids: tuple[str, ...]
    invalid_references: tuple[str, ...]
    error_message: str | None = None


class CitationService:
    """分配请求级来源编号，并校验模型只能引用已提供的来源。"""

    @staticmethod
    def assign_sources(contexts: list[ContextBlock]) -> list[CitationSource]:
        return [
            CitationSource(source_id=f"S{index}", context=context)
            for index, context in enumerate(contexts, start=1)
        ]

    @staticmethod
    def validate(
        answer: str,
        sources: list[CitationSource],
    ) -> CitationValidationResult:
        available_ids = {source.source_id for source in sources}
        cited_ids: list[str] = []
        invalid_references: list[str] = []

        for match in _BRACKET_REFERENCE_PATTERN.finditer(answer):
            reference = match.group(1)
            if _SOURCE_ID_PATTERN.fullmatch(reference) and reference in available_ids:
                if reference not in cited_ids:
                    cited_ids.append(reference)
            else:
                invalid_references.append(match.group(0))

        residual_text = _BRACKET_REFERENCE_PATTERN.sub("", answer)
        if "[" in residual_text or "]" in residual_text:
            invalid_references.append("未闭合或嵌套的方括号")

        if invalid_references:
            return CitationValidationResult(
                is_valid=False,
                cited_source_ids=(),
                invalid_references=tuple(dict.fromkeys(invalid_references)),
                error_message="模型回答包含未授权或格式错误的引用。",
            )
        if not cited_ids:
            return CitationValidationResult(
                is_valid=False,
                cited_source_ids=(),
                invalid_references=(),
                error_message="模型回答没有提供可验证的来源引用。",
            )
        return CitationValidationResult(
            is_valid=True,
            cited_source_ids=tuple(cited_ids),
            invalid_references=(),
        )
