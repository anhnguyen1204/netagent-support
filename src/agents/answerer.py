"""B3: compose grounded answer.

Template-based if NullLLM: "Vấn đề tương tự đã gặp: {problem} -> Giải pháp: {solution}
(nguồn: {source})". LLM-polished if available.
"""
from __future__ import annotations

from src.agents.state import AgentState
from src.kb.schema import ScoredKBEntry
from src.llm.base import LLMClient
from src.llm.null_llm import NullLLM

NO_ANSWER = "Xin lỗi, hiện chưa tìm thấy giải pháp phù hợp trong cơ sở tri thức cho câu hỏi này."

POLISH_PROMPT = """You are a Vietnamese technical support assistant for the netAgent
platform. Using ONLY the verified solution below, write a concise, friendly answer to the
user's question in Vietnamese. Do not invent details beyond the solution. Cite nothing
extra.

User question: {question}
Verified problem: {problem}
Verified solution: {solution}

Answer:"""


def _template_answer(top: ScoredKBEntry) -> str:
    source = top.entry.source_thread_id
    return (
        f"Vấn đề tương tự đã gặp: {top.entry.problem}\n"
        f"→ Giải pháp: {top.entry.solution}\n"
        f"(nguồn: {source})"
    )


def answer(state: AgentState, llm: LLMClient | None) -> AgentState:
    if not state.retrieved:
        state.answer = NO_ANSWER
        return state

    top = state.retrieved[0]

    if llm is not None and not isinstance(llm, NullLLM):
        prompt = POLISH_PROMPT.format(
            question=state.question, problem=top.entry.problem, solution=top.entry.solution
        )
        try:
            polished = llm.complete(prompt).strip()
            state.answer = f"{polished}\n(nguồn: {top.entry.source_thread_id})"
            return state
        except RuntimeError:
            pass  # LLM unavailable -- fall through to template, never a hard dependency

    state.answer = _template_answer(top)
    return state
