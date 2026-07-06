"""B1: KB search + CRAG-style relevance grading.

Calls kb.store.search. If NullLLM (or no LLM), skips relevance grading and thresholds on
raw embedding similarity score. If an LLM is available, grades relevance per result.
"""
from __future__ import annotations

from src.agents.state import AgentState
from src.kb.schema import ScoredKBEntry
from src.kb.store import KBStore
from src.llm.base import LLMClient
from src.llm.null_llm import NullLLM

# Results below this raw cosine similarity are dropped before grading -- clearly
# unrelated matches never make it to the answerer regardless of LLM availability.
MIN_RETRIEVAL_SCORE = 0.35

# A result at or above this cosine similarity is trusted on the embedding signal alone
# and NOT subjected to LLM veto. Small local models (e.g. a 3B) are unreliable binary
# graders and will reject genuinely-relevant entries when the phrasing differs from the
# user's ("mất publish" vs the entry's "sáng bật lại chiều lại tắt"). So the LLM grader
# is used only to filter the ambiguous middle band (MIN_RETRIEVAL_SCORE .. this value),
# where the embedding score is too weak to trust on its own.
LLM_GRADE_TRUST_SCORE = 0.6

GRADE_PROMPT = """You decide if a knowledge-base entry could help answer a user's support
question about the netAgent/netFlow platform. Be lenient: if the entry is about the same
feature, error, or topic — even if the exact wording or symptom differs — it is relevant.
Only answer "no" if the entry is about a clearly different subject.

Answer with ONLY "yes" or "no".

User question: {question}
KB problem: {problem}
KB solution: {solution}

Could this entry help answer the question?"""

REWRITE_PROMPT = """Given a conversation, rewrite the user's latest message into a single
self-contained search query in Vietnamese that includes any context needed from earlier
turns (e.g. resolve "cái đó", "vấn đề thứ hai", "còn nó thì sao"). Respond with ONLY the
rewritten query, nothing else.

Conversation so far:
{history}

Latest message: {question}

Self-contained query:"""


def _rewrite_followup(state: AgentState, llm: LLMClient) -> str:
    """Turn a context-dependent follow-up into a standalone query using history. Returns
    the original question unchanged if there is no history or the rewrite fails."""
    if not state.history:
        return state.question
    history_text = "\n".join(
        f"- User: {t.question}\n  Assistant: {t.answer[:200]}" for t in state.history
    )
    prompt = REWRITE_PROMPT.format(history=history_text, question=state.question)
    try:
        rewritten = llm.complete(prompt).strip()
    except RuntimeError:
        return state.question
    # guard against a model that echoes the prompt or returns something empty/huge
    if not rewritten or len(rewritten) > 300:
        return state.question
    return rewritten


def _llm_grade(question: str, entry: ScoredKBEntry, llm: LLMClient) -> bool:
    prompt = GRADE_PROMPT.format(
        question=question, problem=entry.entry.problem, solution=entry.entry.solution
    )
    try:
        answer = llm.complete(prompt).strip().lower()
    except RuntimeError:
        return True  # LLM unavailable mid-call -- keep the result, threshold already applied
    return answer.startswith("y")


def retrieve(state: AgentState, store: KBStore, llm: LLMClient | None, top_k: int) -> AgentState:
    has_llm = llm is not None and not isinstance(llm, NullLLM)

    # history-aware query rewriting: make a follow-up self-contained before searching, so
    # embedding search finds the right entry ("cái thứ hai" alone matches nothing useful).
    if has_llm and state.history:
        state.search_query = _rewrite_followup(state, llm)
    else:
        state.search_query = state.question
    query = state.search_query

    results = store.search(query, top_k=top_k)
    results = [r for r in results if r.score >= MIN_RETRIEVAL_SCORE]

    if has_llm:
        # keep strong-similarity hits outright; only ask the LLM to adjudicate the
        # weaker middle band, where a wrong-topic false positive is actually likely.
        results = [
            r for r in results if r.score >= LLM_GRADE_TRUST_SCORE or _llm_grade(query, r, llm)
        ]

    state.retrieved = results
    return state
