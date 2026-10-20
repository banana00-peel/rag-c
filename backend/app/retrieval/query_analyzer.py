import re
from dataclasses import dataclass

from app.ingestion.catalog import MODEL_CATALOG, detect_model, normalize_model


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    model: str | None
    intent: str
    variants: tuple[str, ...]


class QueryAnalyzer:
    _intent_patterns = (
        ("配网与 App", ("联网", "配网", "Wi-Fi", "wifi", "APP", "app", "网络")),
        ("故障排查", ("故障", "报错", "无法", "不工作", "停止", "异常", "报警")),
        ("清洁与维护", ("清洗", "清洁", "维护", "滤网", "尘盒", "拖布", "耗材")),
        ("使用说明", ("怎么", "如何", "设置", "使用", "开启", "关闭", "模式")),
        ("售后服务", ("保修", "售后", "维修", "客服", "退换")),
    )

    def analyze(self, question: str, explicit_model: str | None = None) -> QueryPlan:
        normalized = re.sub(r"\s+", " ", question).strip()
        requested_model = explicit_model or detect_model("", normalized)
        detected_model = (
            None
            if not requested_model or requested_model.strip().lower() == "all"
            else normalize_model(requested_model)
        )
        intent = "通用咨询"
        for candidate, keywords in self._intent_patterns:
            if any(keyword.lower() in normalized.lower() for keyword in keywords):
                intent = candidate
                break

        retrieval_query = self._remove_model_alias(normalized, detected_model)
        variants = [retrieval_query or normalized]

        return QueryPlan(
            original=question,
            normalized=normalized,
            model=detected_model,
            intent=intent,
            variants=tuple(dict.fromkeys(variants)),
        )

    @staticmethod
    def _remove_model_alias(question: str, canonical_model: str | None) -> str:
        if not canonical_model:
            return question
        aliases = next(
            (item.aliases for item in MODEL_CATALOG if item.canonical == canonical_model),
            (canonical_model,),
        )
        result = question
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = re.escape(alias).replace(r"\ ", r"\s*")
            result = re.sub(
                rf"(?<![A-Z0-9]){pattern}(?![A-Z0-9])",
                " ",
                result,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\s+", " ", result).strip()

