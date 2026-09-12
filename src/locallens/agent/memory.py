from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class Turn:
    """A single recorded turn in a session's conversation history."""

    query: str
    location: str
    topic: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class _SessionState:
    turns: list[Turn] = field(default_factory=list)
    last_location: str = ""
    last_topic: str = ""


class ConversationMemory:
    """Minimal in-memory, per-session conversation state.

    This gives LocalLens multi-turn context: it remembers the most recently
    established location/topic for a session so a follow-up query such as
    "what about vegan options?" can inherit the city from an earlier turn in
    the same session, without the caller needing to re-specify it.

    This is intentionally a simple in-process store (a dict guarded by a
    lock), not a persistent/distributed cache -- it resets when the process
    restarts. That is sufficient for a single-instance demo/API deployment
    and keeps the "memory" concept easy to reason about and test.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.Lock()

    def record(self, session_id: str, query: str, location: str, topic: str) -> None:
        if not session_id:
            return
        with self._lock:
            state = self._sessions.setdefault(session_id, _SessionState())
            state.turns.append(Turn(query=query, location=location, topic=topic))
            if len(state.turns) > self.max_turns:
                state.turns = state.turns[-self.max_turns :]
            # Sticky: only overwrite when the new turn actually established a
            # value, so an unrelated follow-up doesn't erase known context.
            if location:
                state.last_location = location
            if topic:
                state.last_topic = topic

    def last_location(self, session_id: str) -> str:
        with self._lock:
            state = self._sessions.get(session_id)
            return state.last_location if state else ""

    def last_topic(self, session_id: str) -> str:
        with self._lock:
            state = self._sessions.get(session_id)
            return state.last_topic if state else ""

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            state = self._sessions.get(session_id)
            return list(state.turns) if state else []

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
