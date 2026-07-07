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


# how many of the MOST RECENT prior-turn questions to fold into the embedding search
# (in addition to the first turn, see _anchor_text) to keep a follow-up anchored to the
# conversation's topic.
ANCHOR_RECENT_TURNS = 2


def _entry_id(scored: ScoredKBEntry) -> tuple[str, str]:
    return (scored.entry.source_thread_id, scored.entry.problem)


def _anchor_text(history: list) -> str:
    """Build the context text used to anchor a follow-up's search.

    Always includes the FIRST turn: it states the original problem and carries the
    strongest, most specific signal (e.g. "workflow mất publish"). A later turn like
    "nguyên nhân là gì" is itself generic and carries little topic signal on its own --
    anchoring on only the last N turns can lose the original problem entirely once the
    conversation is a few turns deep, letting a vague follow-up ("cách xử lý lỗi này")
    drift toward a different-but-similar KB entry (e.g. workflow_run vs workflow_publish).
    """
    first = history[0].question
    recent = [t.question for t in history[-ANCHOR_RECENT_TURNS:]]
    # dedupe while preserving order (first may already be in `recent` for short convos)
    seen: set[str] = set()
    parts = []
    for q in [first, *recent]:
        if q not in seen:
            seen.add(q)
            parts.append(q)
    return " ".join(parts)


def retrieve(state: AgentState, store: KBStore, llm: LLMClient | None, top_k: int) -> AgentState:
    has_llm = llm is not None and not isinstance(llm, NullLLM)

    # history-aware query rewriting: make a follow-up self-contained before searching, so
    # embedding search finds the right entry ("cái thứ hai" alone matches nothing useful).
    if has_llm and state.history:
        state.search_query = _rewrite_followup(state, llm)
    else:
        state.search_query = state.question
    query = state.search_query

    if not state.history:
        # first question: plain search, score is the confidence, fast trust-path grading.
        results = [r for r in store.search(query, top_k=top_k) if r.score >= MIN_RETRIEVAL_SCORE]
        if has_llm:
            results = [
                r for r in results if r.score >= LLM_GRADE_TRUST_SCORE or _llm_grade(query, r, llm)
            ]
        state.retrieved = results
        return state

    # Follow-up: separate RECALL from CONFIDENCE.
    #  - Recall: an anchored search (prior questions + this one) reliably finds the
    #    conversation's entry even when a small LLM's rewrite drifts.
    #  - Confidence: score each candidate by how well it matches the BARE query, so an
    #    off-topic follow-up ("thời tiết hôm nay") does NOT inherit the prior topic's
    #    inflated score. The anchor only decides *what* is a candidate, never *how
    #    confident* we are.
    anchor = _anchor_text(state.history)
    anchored_hits = store.search(f"{anchor} {query}", top_k=top_k)

    # Score candidates against BOTH the rewrite and the raw question, taking the max — so
    # a single bad rewrite from a noisy LLM can't sink a follow-up the raw wording would
    # have matched. Both are context-free of the anchor, so off-topic follow-ups still
    # score low and get filtered.
    bare_score: dict[tuple[str, str], float] = {}
    for text in {query, state.question}:
        for r in store.search(text, top_k=top_k):
            key = _entry_id(r)
            bare_score[key] = max(bare_score.get(key, 0.0), r.score)

    results = []
    for hit in anchored_hits:
        hit.score = bare_score.get(_entry_id(hit), 0.0)
        if hit.score >= MIN_RETRIEVAL_SCORE:
            results.append(hit)
    results.sort(key=lambda r: r.score, reverse=True)

    # Follow-ups intentionally do NOT go through the LLM relevance grader. A genuine
    # follow-up phrased in isolation ("tôi nên sửa như thế nào") legitimately scores in
    # the borderline band against a specific KB entry, and a small local model is too
    # unreliable a yes/no judge to gate on. The honest re-score against the bare query
    # already excludes off-topic follow-ups (they fall below MIN_RETRIEVAL_SCORE because
    # the bare query never surfaces the prior topic's entry), so clearing that floor is
    # sufficient to keep the result. This makes multi-turn deterministic instead of
    # flaky at the LLM grader's mercy.
    state.retrieved = results
    return state
