from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.persistence.conversation_repository import ConversationRepository
from app.persistence.database import build_engine, build_session_factory, initialize_database


def build_repository(tmp_path: Path) -> tuple[ConversationRepository, object]:
    engine = build_engine(f"sqlite:///{tmp_path / 'conversation.db'}")
    initialize_database(engine)
    return ConversationRepository(build_session_factory(engine)), engine


def append_turn(
    repository: ConversationRepository,
    session_id: str,
    question: str,
    answer: str,
    *,
    now: datetime,
) -> None:
    repository.append_turn(
        session_id,
        user_content=question,
        standalone_query=question,
        model="G20",
        assistant_content=answer,
        refused=False,
        refusal_code=None,
        ttl_minutes=60,
        now=now,
    )


def test_conversation_history_is_isolated_and_ordered_by_complete_turn(tmp_path: Path):
    repository, engine = build_repository(tmp_path)
    now = datetime.now(timezone.utc)
    try:
        append_turn(repository, "session-a", "第一问", "第一答", now=now)
        append_turn(
            repository,
            "session-a",
            "第二问",
            "第二答",
            now=now + timedelta(seconds=1),
        )
        append_turn(repository, "session-b", "其他问题", "其他回答", now=now)

        messages = repository.load_recent(
            "session-a",
            max_turns=6,
            max_chars=6000,
            ttl_minutes=60,
            now=now + timedelta(seconds=2),
        )

        assert [(message.role, message.content) for message in messages] == [
            ("user", "第一问"),
            ("assistant", "第一答"),
            ("user", "第二问"),
            ("assistant", "第二答"),
        ]
        assert {message.conversation_id for message in messages} == {"session-a"}
    finally:
        engine.dispose()


def test_conversation_history_keeps_only_latest_complete_turns(tmp_path: Path):
    repository, engine = build_repository(tmp_path)
    now = datetime.now(timezone.utc)
    try:
        for index in range(3):
            append_turn(
                repository,
                "session-a",
                f"问题 {index}",
                f"回答 {index}",
                now=now + timedelta(seconds=index),
            )

        messages = repository.load_recent(
            "session-a",
            max_turns=2,
            max_chars=6000,
            ttl_minutes=60,
            now=now + timedelta(seconds=4),
        )

        assert [message.content for message in messages] == [
            "问题 1",
            "回答 1",
            "问题 2",
            "回答 2",
        ]
        assert len({message.turn_id for message in messages}) == 2
    finally:
        engine.dispose()


def test_expired_conversation_does_not_enter_working_memory(tmp_path: Path):
    repository, engine = build_repository(tmp_path)
    now = datetime.now(timezone.utc)
    try:
        append_turn(repository, "session-a", "旧问题", "旧回答", now=now)

        messages = repository.load_recent(
            "session-a",
            max_turns=6,
            max_chars=6000,
            ttl_minutes=60,
            now=now + timedelta(minutes=61),
        )

        assert messages == []
    finally:
        engine.dispose()


def test_latest_turn_is_truncated_without_breaking_turn_pair(tmp_path: Path):
    repository, engine = build_repository(tmp_path)
    now = datetime.now(timezone.utc)
    try:
        append_turn(repository, "session-a", "问题" * 100, "回答" * 300, now=now)

        messages = repository.load_recent(
            "session-a",
            max_turns=6,
            max_chars=120,
            ttl_minutes=60,
            now=now + timedelta(seconds=1),
        )

        assert [message.role for message in messages] == ["user", "assistant"]
        assert sum(len(message.content) for message in messages) <= 120
        assert all(message.content for message in messages)
    finally:
        engine.dispose()
