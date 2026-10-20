from collections import defaultdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import delete, select

from app.domain.models import ConversationMessage, RefusalCode
from app.persistence.database import ConversationEntity, ConversationMessageEntity


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ConversationRepository:
    """持久化有界短期记忆，并保证历史按完整问答轮次返回。"""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def load_recent(
        self,
        conversation_id: str,
        *,
        max_turns: int,
        max_chars: int,
        ttl_minutes: int,
        now: datetime | None = None,
    ) -> list[ConversationMessage]:
        current_time = now or _utc_now()
        with self._session_factory() as session:
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is None:
                return []
            if _aware_utc(conversation.expires_at) <= current_time:
                session.execute(
                    delete(ConversationMessageEntity).where(
                        ConversationMessageEntity.conversation_id == conversation_id
                    )
                )
                conversation.created_at = current_time
                conversation.updated_at = current_time
                conversation.expires_at = current_time + timedelta(minutes=ttl_minutes)
                session.commit()
                return []

            rows = session.scalars(
                select(ConversationMessageEntity)
                .where(ConversationMessageEntity.conversation_id == conversation_id)
                .order_by(ConversationMessageEntity.created_at.desc())
                .limit(max_turns * 2)
            ).all()
            return self._select_complete_turns(rows, max_turns=max_turns, max_chars=max_chars)

    def append_turn(
        self,
        conversation_id: str,
        *,
        user_content: str,
        standalone_query: str,
        model: str | None,
        assistant_content: str,
        refused: bool,
        refusal_code: RefusalCode | None,
        ttl_minutes: int,
        now: datetime | None = None,
    ) -> None:
        current_time = now or _utc_now()
        turn_id = str(uuid4())
        with self._session_factory() as session:
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is None:
                conversation = ConversationEntity(
                    id=conversation_id,
                    created_at=current_time,
                    updated_at=current_time,
                    expires_at=current_time + timedelta(minutes=ttl_minutes),
                )
                session.add(conversation)
            else:
                conversation.updated_at = current_time
                conversation.expires_at = current_time + timedelta(minutes=ttl_minutes)

            session.add_all(
                [
                    ConversationMessageEntity(
                        id=str(uuid4()),
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        role="user",
                        content=user_content,
                        standalone_query=standalone_query,
                        model=model,
                        created_at=current_time,
                    ),
                    ConversationMessageEntity(
                        id=str(uuid4()),
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        role="assistant",
                        content=assistant_content,
                        refused=refused,
                        refusal_code=str(refusal_code) if refusal_code is not None else None,
                        created_at=current_time + timedelta(microseconds=1),
                    ),
                ]
            )
            session.commit()

    def delete_expired(self, now: datetime | None = None) -> int:
        current_time = now or _utc_now()
        with self._session_factory() as session:
            expired_ids = [
                conversation.id
                for conversation in session.scalars(select(ConversationEntity)).all()
                if _aware_utc(conversation.expires_at) <= current_time
            ]
            if not expired_ids:
                return 0
            session.execute(
                delete(ConversationMessageEntity).where(
                    ConversationMessageEntity.conversation_id.in_(expired_ids)
                )
            )
            session.execute(
                delete(ConversationEntity).where(ConversationEntity.id.in_(expired_ids))
            )
            session.commit()
            return len(expired_ids)

    @staticmethod
    def _select_complete_turns(
        rows: list[ConversationMessageEntity],
        *,
        max_turns: int,
        max_chars: int,
    ) -> list[ConversationMessage]:
        grouped: dict[str, list[ConversationMessageEntity]] = defaultdict(list)
        for row in rows:
            grouped[row.turn_id].append(row)

        complete_turns = [
            sorted(messages, key=lambda message: message.created_at)
            for messages in grouped.values()
            if {message.role for message in messages} == {"user", "assistant"}
        ]
        complete_turns.sort(key=lambda messages: messages[0].created_at, reverse=True)

        selected: list[list[tuple[ConversationMessageEntity, str]]] = []
        character_count = 0
        for turn in complete_turns[:max_turns]:
            turn_chars = sum(len(message.content) for message in turn)
            if selected and character_count + turn_chars > max_chars:
                break
            if not selected and turn_chars > max_chars:
                user = next(message for message in turn if message.role == "user")
                assistant = next(message for message in turn if message.role == "assistant")
                user_budget = min(len(user.content), max(1, max_chars // 3))
                assistant_budget = max_chars - user_budget
                selected.append(
                    [
                        (user, ConversationRepository._truncate(user.content, user_budget)),
                        (
                            assistant,
                            ConversationRepository._truncate(
                                assistant.content,
                                assistant_budget,
                            ),
                        ),
                    ]
                )
                break
            selected.append([(message, message.content) for message in turn])
            character_count += turn_chars

        ordered_rows = [item for turn in reversed(selected) for item in turn]
        return [ConversationRepository._to_message(row, content) for row, content in ordered_rows]

    @staticmethod
    def _truncate(content: str, budget: int) -> str:
        if len(content) <= budget:
            return content
        if budget <= 1:
            return content[:budget]
        prefix_length = max(1, (budget - 1) * 2 // 3)
        suffix_length = budget - prefix_length - 1
        if not suffix_length:
            return content[:budget]
        return content[:prefix_length] + "…" + content[-suffix_length:]

    @staticmethod
    def _to_message(entity: ConversationMessageEntity, content: str) -> ConversationMessage:
        return ConversationMessage(
            id=entity.id,
            conversation_id=entity.conversation_id,
            turn_id=entity.turn_id,
            role=entity.role,
            content=content,
            standalone_query=entity.standalone_query,
            model=entity.model,
            refused=entity.refused,
            refusal_code=(
                RefusalCode(entity.refusal_code) if entity.refusal_code is not None else None
            ),
            created_at=entity.created_at,
        )
