"""Conversation memory for RactoMail."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._models import ConversationTurn, EmailRef, MailFacets, MailResponse

if TYPE_CHECKING:
    from .kit import RactoMailKit


@dataclass
class SessionStore:
    """Small in-memory session store used by default."""

    _sessions: dict[str, ConversationSession] = field(default_factory=dict)

    def save(self, session: ConversationSession) -> None:
        """Persist the latest session state."""
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> ConversationSession | None:
        """Return an existing session if present."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        """Delete a finished session."""
        self._sessions.pop(session_id, None)

    def count(self) -> int:
        """Return the number of live sessions."""
        return len(self._sessions)


class ConversationSession:
    """Stateful query wrapper over :class:`RactoMailKit`."""

    def __init__(
        self,
        *,
        session_id: str,
        user: str,
        mail: RactoMailKit,
        context: str | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.session_id = session_id
        self.user = user
        self.context = context
        self._mail = mail
        self._store = store
        self.active_entity: str | None = None
        self.active_filters = MailFacets()
        self.active_result_set: list[EmailRef] | None = None
        self.topic_stack: list[str] = []
        self.turn_history: list[ConversationTurn] = []

    def ask(self, query: str) -> MailResponse:
        """Run a stateful synchronous mail query."""
        resolved_query, message_ids = self._resolve_query(query)
        facets = self.active_filters.model_copy(deep=True)
        if message_ids is not None:
            facets.message_ids = message_ids
        response = self._mail.ask(resolved_query, facets=facets)
        self._record_turn(query, resolved_query, response, facets)
        return response

    async def aask(self, query: str) -> MailResponse:
        """Async variant of :meth:`ask`."""
        resolved_query, message_ids = self._resolve_query(query)
        facets = self.active_filters.model_copy(deep=True)
        if message_ids is not None:
            facets.message_ids = message_ids
        response = await self._mail.aask(resolved_query, facets=facets)
        self._record_turn(query, resolved_query, response, facets)
        return response

    def end(self) -> None:
        """Persist or clear the session when the caller is done."""
        if self._store is not None:
            self._store.save(self)

    def _resolve_query(self, query: str) -> tuple[str, list[str] | None]:
        lowered = query.casefold()
        resolved = query
        if self.active_entity is not None and any(
            marker in lowered for marker in ("same vendor", "same sender", "same contact")
        ):
            resolved = f"{query} {self.active_entity}"

        if self.active_result_set and any(
            marker in lowered
            for marker in ("those", "that", "them", "these", "only the", "filter to")
        ):
            return resolved, [reference.message_id for reference in self.active_result_set]
        return resolved, None

    def _record_turn(
        self,
        query_original: str,
        query_resolved: str,
        response: MailResponse,
        facets: MailFacets,
    ) -> None:
        self.active_result_set = response.references or None
        if response.references:
            self.active_entity = response.references[0].sender
        self.topic_stack.append(query_original)
        self.turn_history.append(
            ConversationTurn(
                turn_number=len(self.turn_history) + 1,
                query_original=query_original,
                query_resolved=query_resolved,
                filters_applied=facets,
                answer=response.answer,
                references=response.references,
            )
        )
        if self._store is not None:
            self._store.save(self)

