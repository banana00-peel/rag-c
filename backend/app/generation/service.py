import logging
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import Settings
from app.domain.models import CitationSource
from app.generation.prompts import SYSTEM_PROMPT, build_context
from app.generation.schemas import GenerationResult

logger = logging.getLogger(__name__)


class GenerationOutputError(RuntimeError):
    """生成模型未遵守结构化输出协议。"""


# 接口，流程图依赖，不强制使用模型实现类，可替换其他模型实现类
class AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        sources: Sequence[CitationSource],
    ) -> GenerationResult: ...


# 实现类，Deepseek的具体实现
class DeepSeekStructuredGenerator:
    """通过 JSON mode 获取结果，再由 Pydantic 严格校验业务协议。"""

    def __init__(self, settings: Settings):
        llm = ChatOpenAI(
            api_key=settings.deepseek_api_key or "deepseek-key-not-configured",
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0,
            max_retries=2,
        )
        self._structured_llm = llm.with_structured_output(
            GenerationResult,
            method="json_mode",
        )

    def generate(
        self,
        question: str,
        sources: Sequence[CitationSource],
    ) -> GenerationResult:
        prompt = SYSTEM_PROMPT.format(context=build_context(sources))
        try:
            result = self._structured_llm.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=question),
                ]
            )
        except Exception as exc:
            logger.warning("生成模型结构化输出失败：%s", type(exc).__name__)
            raise GenerationOutputError("生成模型未返回符合约定的结构化结果。") from exc

        if not isinstance(result, GenerationResult):
            logger.warning("生成模型返回了非预期结果类型：%s", type(result).__name__)
            raise GenerationOutputError("生成模型未返回符合约定的结构化结果。")
        return result
