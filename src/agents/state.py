"""Shared graph state schema for the LangGraph agent graph (Regime B)."""
from __future__ import annotations

from pydantic import BaseModel

from src.kb.schema import ScoredKBEntry


class AgentState(BaseModel):
    question: str
    retrieved: list[ScoredKBEntry] = []
    answer: str | None = None
    confidence: float | None = None
    decision: str | None = None  # auto_reply | suggest_to_staff | escalate