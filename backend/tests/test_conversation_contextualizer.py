from datetime import datetime, timezone

from app.conversation.contextualizer import DeepSeekQuestionContextualizer
from app.conversation.schemas import ContextualizedQuery
from app.domain.models import ConversationMessage


class FakeStructuredModel:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return self._result


def history_message(role: str, content: str, model: str | None = "G20"):
    return ConversationMessage(
        id=f"message-{role}",
        conversation_id="session-a",
        turn_id="turn-a",
        role=role,
        content=content,
        standalone_query=content if role == "user" else None,
        model=model if role == "user" else None,
        refused=False if role == "assistant" else None,
        refusal_code=None,
        created_at=datetime.now(timezone.utc),
    )


def build_contextualizer(model: FakeStructuredModel) -> DeepSeekQuestionContextualizer:
    contextualizer = object.__new__(DeepSeekQuestionContextualizer)
    contextualizer._structured_llm = model
    return contextualizer


def test_first_turn_skips_contextualization_model_call():
    model = FakeStructuredModel(error=AssertionError("首轮不应调用改写模型"))
    contextualizer = build_contextualizer(model)

    result = contextualizer.contextualize("G20 如何清理尘盒？", None, [])

    assert result.standalone_query == "G20 如何清理尘盒？"
    assert result.model == "G20"
    assert result.history_used is False
    assert model.call_count == 0


def test_follow_up_is_rewritten_as_standalone_query():
    model = FakeStructuredModel(
        ContextualizedQuery(
            standalone_query="G20 的滤网应该如何清理？",
            model="G20",
            history_used=True,
        )
    )
    contextualizer = build_contextualizer(model)
    history = [
        history_message("user", "G20 如何清理尘盒？"),
        history_message("assistant", "请取出尘盒后清理。"),
    ]

    result = contextualizer.contextualize("那滤网呢？", None, history)

    assert result.standalone_query == "G20 的滤网应该如何清理？"
    assert result.model == "G20"
    assert result.history_used is True


def test_current_question_model_overrides_historical_model():
    model = FakeStructuredModel(
        ContextualizedQuery(
            standalone_query="S5 的滤网应该如何清理？",
            model="G20",
            history_used=True,
        )
    )
    contextualizer = build_contextualizer(model)

    result = contextualizer.contextualize(
        "那 S5 呢？",
        None,
        [history_message("user", "G20 如何清理尘盒？")],
    )

    assert result.model == "S5"


def test_contextualization_failure_falls_back_without_blocking_rag():
    contextualizer = build_contextualizer(FakeStructuredModel(error=RuntimeError("上游失败")))

    result = contextualizer.contextualize(
        "那滤网呢？",
        None,
        [history_message("user", "G20 如何清理尘盒？")],
    )

    assert result.standalone_query == "那滤网呢？"
    assert result.model == "G20"
    assert result.history_used is True
