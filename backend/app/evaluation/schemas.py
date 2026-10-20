import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.models import RefusalCode


class EvaluationDataError(ValueError):
    pass


class EvaluationCase(BaseModel):
    """一条人工标注的评测样本，页码使用 PDF 物理页码。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    split: Literal["calibration", "test"]
    question: str = Field(min_length=2, max_length=500)
    model: str | None = Field(default=None, max_length=100)
    category: str = Field(min_length=2, max_length=100)
    expected_behavior: Literal["answer", "refuse"]
    retrieval_expectation: Literal["answer", "refuse", "retrieve_for_refusal"] | None = None
    document_sha256: str | None = None
    gold_pages: list[int] = Field(default_factory=list)
    gold_evidence: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    allowed_refusal_codes: list[RefusalCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ground_truth(self):
        if self.retrieval_expectation is None:
            self.retrieval_expectation = self.expected_behavior
        if self.expected_behavior == "answer":
            if self.retrieval_expectation != "answer":
                raise ValueError("可回答样本的检索预期必须为 answer。")
            if not self.document_sha256 or not re.fullmatch(
                r"[0-9a-f]{64}", self.document_sha256
            ):
                raise ValueError("可回答样本必须提供有效的文档 SHA-256。")
            if not self.gold_pages or any(page < 1 for page in self.gold_pages):
                raise ValueError("可回答样本必须提供有效的正确页码。")
            if not self.gold_evidence:
                raise ValueError("可回答样本必须提供人工核对过的证据文本。")
            if not self.expected_keywords:
                raise ValueError("可回答样本必须提供关键事实关键词。")
        elif not self.allowed_refusal_codes:
            raise ValueError("拒答样本必须声明允许的拒答码。")
        return self


def load_dataset(path: str | Path) -> list[EvaluationCase]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise EvaluationDataError(f"评测集不存在：{dataset_path}")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            case = EvaluationCase.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise EvaluationDataError(
                f"评测集第 {line_number} 行格式错误：{exc}"
            ) from exc
        if case.id in seen_ids:
            raise EvaluationDataError(f"评测样本 ID 重复：{case.id}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise EvaluationDataError("评测集不能为空。")
    if not any(case.split == "calibration" for case in cases):
        raise EvaluationDataError("评测集必须包含 calibration 样本。")
    if not any(case.split == "test" for case in cases):
        raise EvaluationDataError("评测集必须包含 test 样本。")
    return cases
