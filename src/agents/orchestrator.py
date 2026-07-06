"""B0: parse + route. Entry node of the agent graph."""
from __future__ import annotations

from src.agents.state import AgentState


def orchestrate(state: AgentState) -> AgentState:
    """Normalize the incoming question. This is the seam where language detection,
    routing to different retrieval strategies, or query rewriting would live; for now
    it just trims whitespace and passes the question through to retrieval.
    """
    state.question = state.question.strip()
    return state
