import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config.settings import Settings
from app.domain.models import CitationSource, ContextBlock, RefusalCode, RetrievedChunk
from app.generation.citation_service import CitationService
from app.generation.schemas import GenerationResult
from app.generation.service import AnswerGenerator, DeepSeekStructuredGenerator
from app.retrieval.context_selector import ContextSelector
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    question: str
    explicit_model: str | None
    plan: object
    candidates: list[RetrievedChunk]
    contexts: list[ContextBlock]
    citation_sources: list[CitationSource]
    generation_result: GenerationResult
    cited_source_ids: list[str]
    citation_valid: bool
    refused: bool
    refusal_code: RefusalCode | None
    refusal_reason: str | None
    answer: str


class ChatGraphService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: AnswerGenerator | None = None,
    ):
        self._settings = settings
        self._analyzer = QueryAnalyzer()
        self._retriever = retriever
        self._context_selector = ContextSelector(settings)
        self._citation_service = CitationService()
        self._generator = generator or DeepSeekStructuredGenerator(settings)
        self._graph = self._build_graph()

    def answer(self, question: str, explicit_model: str | None = None) -> ChatState:
        return self._graph.invoke(
            {
                "question": question,
                "explicit_model": explicit_model,
            }
        )

    def _build_graph(self):
        workflow = StateGraph(ChatState)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("gate", self._gate)
        workflow.add_node("prepare_sources", self._prepare_sources)
        workflow.add_node("generate", self._generate)
        workflow.add_node("refuse_insufficient_evidence", self._refuse_insufficient_evidence)
        workflow.add_node("validate_citations", self._validate_citations)
        workflow.add_node("refuse", self._refuse)
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "gate")
        workflow.add_conditional_edges(
            "gate",
            self._route,
            {
                "generate": "prepare_sources",
                "refuse": "refuse",
            },
        )
        workflow.add_edge("prepare_sources", "generate")
        workflow.add_conditional_edges(
            "generate",
            self._route_generation,
            {
                "answer": "validate_citations",
                "refuse": "refuse_insufficient_evidence",
            },
        )
        workflow.add_edge("refuse_insufficient_evidence", END)
        workflow.add_edge("validate_citations", END)
        workflow.add_edge("refuse", END)
        return workflow.compile()

    def _analyze(self, state: ChatState) -> dict:
        plan = self._analyzer.analyze(state["question"], state.get("explicit_model"))
        return {"plan": plan}

    def _retrieve(self, state: ChatState) -> dict:
        plan = state["plan"]
        candidates = self._retriever.retrieve(plan)
        return {"candidates": candidates}

    def _gate(self, state: ChatState) -> dict:
        candidates = state.get("candidates", [])
        if not candidates:
            return {
                "refused": True,
                "refusal_code": RefusalCode.NO_RECALL,
                "refusal_reason": "知识库没有召回相关内容。",
            }
        relevant_candidates = self._retriever.filter_relevant(candidates)
        if not relevant_candidates:
            return {
                "refused": True,
                "refusal_code": RefusalCode.LOW_RELEVANCE,
                "refusal_reason": "召回内容的相似度低于当前阈值。",
            }
        contexts = self._context_selector.select(relevant_candidates)
        return {
            "refused": False,
            "refusal_code": None,
            "refusal_reason": None,
            "contexts": contexts,
        }

    @staticmethod
    def _route(state: ChatState) -> str:
        return "refuse" if state.get("refused") else "generate"

    def _prepare_sources(self, state: ChatState) -> dict:
        sources = self._citation_service.assign_sources(state["contexts"])
        return {"citation_sources": sources}

    def _generate(self, state: ChatState) -> dict:
        result = self._generator.generate(state["question"], state["citation_sources"])
        return {"generation_result": result}

    @staticmethod
    def _route_generation(state: ChatState) -> str:
        return state["generation_result"].outcome

    @staticmethod
    def _refuse_insufficient_evidence(state: ChatState) -> dict:
        return {
            "answer": (
                "检索到了相关内容，但当前知识库没有足够证据回答该问题，"
                "建议联系石头官方人工客服。"
            ),
            "refused": True,
            "refusal_code": RefusalCode.INSUFFICIENT_EVIDENCE,
            "refusal_reason": "召回内容与问题相关，但缺少回答所需的直接证据。",
            "cited_source_ids": [],
        }

    def _validate_citations(self, state: ChatState) -> dict:
        generation_result = state["generation_result"]
        validation = self._citation_service.validate(
            generation_result.answer or "",
            state.get("citation_sources", []),
        )
        if validation.is_valid:
            return {
                "answer": generation_result.answer or "",
                "refused": False,
                "refusal_code": None,
                "refusal_reason": None,
                "citation_valid": True,
                "cited_source_ids": list(validation.cited_source_ids),
            }

        logger.warning(
            "模型回答引用校验失败：%s；非法引用：%s",
            validation.error_message,
            "、".join(validation.invalid_references) or "无",
        )
        return {
            "answer": "生成的回答未通过来源引用校验，暂时无法提供可靠答案，请稍后重试或联系石头官方人工客服。",
            "refused": True,
            "refusal_code": RefusalCode.INVALID_CITATION,
            "refusal_reason": "模型生成内容的来源引用无法由后端验证。",
            "citation_valid": False,
            "cited_source_ids": [],
        }

    @staticmethod
    def _refuse(state: ChatState) -> dict:
        if state.get("refusal_code") == RefusalCode.LOW_RELEVANCE:
            answer = "检索到了可能相关的内容，但相似度不足，建议联系石头官方人工客服。"
        else:
            answer = "当前知识库未找到与问题匹配的内容，建议联系石头官方人工客服。"
        return {
            "answer": answer,
        }

