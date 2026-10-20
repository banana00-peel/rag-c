from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationResult(BaseModel):
    """生成模型的受控输出，避免从自然语言推断回答或拒答状态。"""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["answer", "refuse"]
    answer: str | None = Field(default=None, max_length=8000)
    refusal_code: Literal["insufficient_evidence"] | None = None

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.outcome == "answer":
            if not self.answer or not self.answer.strip():
                raise ValueError("回答结果必须包含非空回答正文。")
            if self.refusal_code is not None:
                raise ValueError("回答结果不能包含拒答码。")
            self.answer = self.answer.strip()
            return self

        if self.answer is not None:
            raise ValueError("拒答结果不能包含回答正文。")
        if self.refusal_code != "insufficient_evidence":
            raise ValueError("生成阶段只能返回证据不足拒答码。")
        return self
