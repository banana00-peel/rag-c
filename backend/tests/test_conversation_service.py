from types import SimpleNamespace

from app.config.settings import Settings
from app.conversation.schemas import ContextualizedQuery
from app.services.conversation_service import ConversationService


class FakeConversationRepository:
    def __init__(self, history=None):
        self.history = history or []
        self.appended = None

    def load_recent(self, session_id: str, **kwargs):
        return self.history

    def append_turn(self, session_id: str, **kwargs):
        self.appended = (session_id, kwargs)


class FakeContextualizer:
    def contextualize(self, question, explicit_model, history):
        return ContextualizedQuery(
            standalone_query="G20 的滤网应该如何清理？",
            model="G20",
            history_used=True,
        )


class FakeChat:
    def __init__(self):
        self.received = None

    def answer(self, question: str, explicit_model: str | None):
        self.received = (question, explicit_model)
        return {
            "answer": "使用清水清洗滤网。[S1]",
            "refused": False,
            "refusal_code": None,
            "plan": SimpleNamespace(model="G20"),
        }


def test_conversation_service_retrieves_with_rewritten_query_and_saves_turn():
    repository = FakeConversationRepository()
    chat = FakeChat()
    service = ConversationService(
        Settings(),
        repository,
        FakeContextualizer(),
        chat,
    )

    result = service.answer("session-a", "那滤网呢？")

    assert chat.received == ("G20 的滤网应该如何清理？", "G20")
    assert result["history_used"] is True
    assert result["standalone_query"] == "G20 的滤网应该如何清理？"
    assert result["session_id"] == "session-a"
    assert repository.appended[0] == "session-a"
    assert repository.appended[1]["user_content"] == "那滤网呢？"
    assert repository.appended[1]["model"] == "G20"
