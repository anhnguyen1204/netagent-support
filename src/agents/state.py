"""Shared graph state schema for the LangGraph agent graph (Regime B)."""
from __future__ import annotations

from pydantic import BaseModel

from src.kb.schema import ScoredKBEntry


class Turn(BaseModel):
    """One prior exchange in a conversation."""

    question: str
    answer: str


class AgentState(BaseModel):
    question: str
    # prior turns in this conversation (oldest first), empty for a fresh/stateless ask
    history: list[Turn] = []
    # set by orchestrate: new_problem | follow_up | chit_chat | off_topic
    turn_type: str | None = None
    # question after history-aware rewriting (follow-ups made self-contained), set by
    # orchestrate; falls back to `question` when there's nothing to rewrite
    search_query: str | None = None
    retrieved: list[ScoredKBEntry] = []
    answer: str | None = None
    confidence: float | None = None
    # auto_reply | suggest_to_staff | escalate | direct_reply (chit_chat/off_topic,
    # never touches the KB or the alerter)
    decision: str | None = None
