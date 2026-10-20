import logging

from app.config.settings import Settings
from app.conversation.contextualizer import QuestionContextualizer
from app.domain.models import RefusalCode
from app.generation.chat_graph import ChatGraphService
from app.persistence.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


class ConversationService:
    """编排短期记忆、独立问题改写和现有可信 RAG 链路。"""

    def __init__(
        self,
        settings: Settings,
        conversations: ConversationRepository,
        contextualizer: QuestionContextualizer,
        chat: ChatGraphService,
    ):
        self._settings = settings
        self._conversations = conversations
        self._contextualizer = contextualizer
        self._chat = chat

    def answer(
        self,
        session_id: str,
        question: str,
        explicit_model: str | None = None,
    ) -> dict:
        history = self._conversations.load_recent(
            session_id,
            max_turns=self._settings.conversation_history_turns,
            max_chars=self._settings.conversation_history_max_chars,
            ttl_minutes=self._settings.conversation_ttl_minutes,
        )
        contextualized = self._contextualizer.contextualize(
            question,
            explicit_model,
            history,
        )
        result = self._chat.answer(
            contextualized.standalone_query,
            contextualized.model,
        )
        plan = result.get("plan")
        resolved_model = getattr(plan, "model", contextualized.model)
        refusal_code = result.get("refusal_code")
        self._conversations.append_turn(
            session_id,
            user_content=question,
            standalone_query=contextualized.standalone_query,
            model=resolved_model,
            assistant_content=result.get("answer", ""),
            refused=bool(result.get("refused", False)),
            refusal_code=(
                refusal_code
                if isinstance(refusal_code, RefusalCode)
                else RefusalCode(refusal_code) if refusal_code else None
            ),
            ttl_minutes=self._settings.conversation_ttl_minutes,
        )
        result["session_id"] = session_id
        result["standalone_query"] = contextualized.standalone_query
        result["history_used"] = contextualized.history_used
        logger.info(
            "会话问答完成：session_id=%s history_used=%s model=%s refused=%s",
            session_id,
            contextualized.history_used,
            resolved_model or "全部",
            result.get("refused", False),
        )
        return result
