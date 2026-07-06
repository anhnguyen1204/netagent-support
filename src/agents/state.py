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
    # question after optional history-aware rewriting (follow-ups made self-contained);
    # falls back to `question` when no rewrite happens
    search_query: str | None = None
    retrieved: list[ScoredKBEntry] = []
    answer: str | None = None
    confidence: float | None = None
    decision: str | None = None  # auto_reply | suggest_to_staff | escalate
