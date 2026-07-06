"""Per-session conversation memory.

An in-process store mapping session_id -> recent turns, so multi-turn `/ask` can carry
context (a follow-up like "và cái thứ hai thì sao?" is meaningless without the prior
turn). In-memory is intentional for this local/demo scope — a real deployment would back
this with Redis or a DB behind the same tiny interface.
"""
from __future__ import annotations

from collections import OrderedDict

from src.agents.state import Turn

# how many recent turns to keep and feed back as context per session
MAX_TURNS_PER_SESSION = 6
# cap on simultaneously tracked sessions (evict least-recently-used beyond this)
MAX_SESSIONS = 1000


class ConversationStore:
    def __init__(self, max_turns: int = MAX_TURNS_PER_SESSION, max_sessions: int = MAX_SESSIONS):
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[Turn]] = OrderedDict()

    def history(self, session_id: str | None) -> list[Turn]:
        if not session_id or session_id not in self._sessions:
            return []
        self._sessions.move_to_end(session_id)  # mark recently used
        return list(self._sessions[session_id])

    def append(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        turns = self._sessions.get(session_id, [])
        turns.append(Turn(question=question, answer=answer))
        self._sessions[session_id] = turns[-self.max_turns :]
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)  # evict least-recently-used

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
