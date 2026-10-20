from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router


class FakeConversationService:
    def __init__(self):
        self.received = None

    def answer(self, session_id: str, question: str, model: str | None):
        self.received = (session_id, question, model)
        return {
            "answer": "当前知识库没有足够证据回答该问题。",
            "refused": True,
            "refusal_code": "insufficient_evidence",
            "refusal_reason": "缺少直接证据。",
            "citation_sources": [],
            "cited_source_ids": [],
            "standalone_query": "G20 清水箱容量是多少？",
            "history_used": True,
        }


def build_client() -> tuple[TestClient, FakeConversationService]:
    app = FastAPI()
    conversation = FakeConversationService()
    app.state.container = SimpleNamespace(conversation=conversation)
    app.include_router(router, prefix="/api")
    return TestClient(app), conversation


def test_chat_api_uses_session_and_returns_memory_trace():
    client, conversation = build_client()
    session_id = str(uuid4())

    response = client.post(
        "/api/chat",
        json={"query": "那容量呢？", "model": "G20", "session_id": session_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["standalone_query"] == "G20 清水箱容量是多少？"
    assert payload["history_used"] is True
    assert conversation.received == (session_id, "那容量呢？", "G20")


def test_chat_api_generates_session_id_when_missing():
    client, conversation = build_client()

    response = client.post("/api/chat", json={"query": "G20 如何清理尘盒？"})

    assert response.status_code == 200
    generated_id = response.json()["session_id"]
    assert str(UUID(generated_id)) == generated_id
    assert conversation.received[0] == generated_id


def test_chat_api_rejects_invalid_session_id():
    client, _ = build_client()

    response = client.post(
        "/api/chat",
        json={"query": "G20 如何清理尘盒？", "session_id": "not-a-uuid"},
    )

    assert response.status_code == 422
