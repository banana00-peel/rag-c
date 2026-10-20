from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextualizedQuery(BaseModel):
    """将依赖历史的追问转换为可独立检索的问题。"""

    model_config = ConfigDict(extra="forbid")

    standalone_query: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(default=None, max_length=100)
    history_used: bool

    @model_validator(mode="after")
    def normalize_values(self):
        self.standalone_query = self.standalone_query.strip()
        if not self.standalone_query:
            raise ValueError("独立检索问题不能为空。")
        if self.model is not None:
            self.model = self.model.strip() or None
        return self
