"""B4: confidence scoring + gate.

Confidence = top retrieval score x the matched entry's stored confidence. Applies gate
thresholds from src.config.GateThresholds to pick the decision:
    auto_reply | suggest_to_staff | escalate
"""
from __future__ import annotations

from src.agents.state import AgentState
from src.config import GateThresholds
from src.llm.base import LLMClient


def critique(state: AgentState, llm: LLMClient | None, thresholds: GateThresholds) -> AgentState:
    if not state.retrieved:
        state.confidence = 0.0
        state.decision = "escalate"
        return state

    top = state.retrieved[0]
    # Combine how well the query matched (retrieval score) with how trustworthy the
    # matched entry itself is (its stored confidence). A strong match to a low-confidence
    # entry should not auto-reply. Both are 0-1, so their product stays 0-1.
    confidence = float(top.score) * float(top.entry.confidence)
    state.confidence = confidence

    if confidence >= thresholds.auto_reply_min:
        state.decision = "auto_reply"
    elif confidence >= thresholds.suggest_to_staff_min:
        state.decision = "suggest_to_staff"
    else:
        state.decision = "escalate"

    return state
