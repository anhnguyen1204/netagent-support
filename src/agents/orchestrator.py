"""B0: parse + route. Entry node of the agent graph.

Classifies the incoming message's turn_type with a single LLM call, and — for
new_problem/follow_up — rewrites it into a self-contained search query in the same
call (folds what used to be two separate LLM calls, classify + rewrite, into one).

turn_type values:
    new_problem  -- a fresh technical question/report, no dependency on prior turns.
    follow_up    -- refers back to the current conversation ("nguyên nhân là gì",
                    "còn cái kia thì sao").
    chit_chat    -- greeting/thanks/meta ("hi", "cảm ơn", "bạn là ai").
    off_topic    -- clearly unrelated to netAgent/netFlow support.
"""
from __future__ import annotations

import json

from src.agents.state import AgentState
from src.llm.base import LLMClient

VALID_TURN_TYPES = {"new_problem", "follow_up", "chit_chat", "off_topic"}

# how many of the most recent prior turns to show in the classification prompt, in
# addition to the first turn -- mirrors retrieval.py's _anchor_text reasoning: the
# first turn carries the strongest topic signal and must not be dropped as a
# conversation grows, or a generic later follow-up loses its anchor.
RECENT_TURNS_IN_PROMPT = 2

CLASSIFY_PROMPT = """You triage one message in a Vietnamese technical support chat for
the netAgent/netFlow platform. Decide its turn_type:

- "new_problem": a fresh technical question or problem report, understandable on its
  own without needing earlier conversation.
- "follow_up": refers back to the current conversation (e.g. "nguyên nhân là gì",
  "còn cái kia thì sao", "làm sao để sửa nó") -- meaningless without the prior turns.
- "chit_chat": a greeting, thanks, or meta question about the bot itself ("hi", "chào
  bạn", "cảm ơn nhé", "bạn là ai").
- "off_topic": clearly unrelated to netAgent/netFlow support (weather, food, general
  knowledge, etc).

If turn_type is "new_problem" or "follow_up", also produce "search_query": a single
self-contained Vietnamese search query. For "follow_up", resolve references using the
conversation so it stands alone (e.g. "cái đó" -> what it refers to). For
"new_problem", this is just the message itself, cleaned up if needed.
For "chit_chat"/"off_topic", set "search_query" to null.

Respond with ONLY a JSON object: {{"turn_type": "...", "search_query": "..." or null}}

{history_block}
Latest message: {question}

JSON:"""


def _history_block(state: AgentState) -> str:
    if not state.history:
        return ""
    first = state.history[0]
    recent = state.history[-RECENT_TURNS_IN_PROMPT:]
    seen_idx = {0} if state.history[0] is first else set()
    lines = [f"- User: {first.question}\n  Assistant: {first.answer[:200]}"]
    for i, t in enumerate(state.history):
        if t in recent and i not in seen_idx:
            lines.append(f"- User: {t.question}\n  Assistant: {t.answer[:200]}")
            seen_idx.add(i)
    return "Earlier conversation:\n" + "\n".join(lines) + "\n"


def _parse_classification(raw: str, question: str) -> tuple[str, str]:
    """Returns (turn_type, search_query). Falls back to the safest existing behavior
    (treat as a new_problem, search on the raw question) on any parse failure, rather
    than guessing chit_chat/off_topic and silently dropping a real question."""
    try:
        data = json.loads(raw.strip())
        turn_type = data.get("turn_type")
        if turn_type not in VALID_TURN_TYPES:
            return "new_problem", question
        search_query = data.get("search_query") or question
        return turn_type, search_query
    except (json.JSONDecodeError, AttributeError, TypeError):
        return "new_problem", question


def orchestrate(state: AgentState, llm: LLMClient) -> AgentState:
    state.question = state.question.strip()

    prompt = CLASSIFY_PROMPT.format(
        history_block=_history_block(state),
        question=state.question,
    )
    try:
        raw = llm.complete(prompt)
        turn_type, search_query = _parse_classification(raw, state.question)
    except RuntimeError:
        turn_type, search_query = "new_problem", state.question

    state.turn_type = turn_type
    state.search_query = search_query
    return state
