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

POLISH_PROMPT = """You are answering a Vietnamese technical support question for the
netAgent platform, inside a chat UI where the answer is shown directly (there is no
separate greeting/sign-off from a human agent around it).

Using ONLY the verified solution below, write the answer in Vietnamese. Hard rules:
- Start directly with the answer content. NEVER open with "Chào bạn", "Xin chào", or any
  greeting.
- NEVER close with a sign-off, well-wish, or offer of further help ("Chúc bạn...",
  "Nếu cần hỗ trợ thêm...", "Trân trọng", etc).
- Never invent a NEW fact, cause, or fix step that is not stated or clearly implied by
  the verified solution.
- If the user asks for an example / cách làm cụ thể / minh hoạ: you MAY illustrate the
  verified solution with a concrete instance (e.g. turn its abstract instruction into one
  worked scenario) as long as every detail in the example is a direct application of the
  solution, not a new fact. If the solution is too abstract to illustrate this way without
  inventing something, say plainly that no worked example is available and restate the
  actionable step instead of repeating the same explanation verbatim.
- Do not just repeat your previous answer word-for-word if the user is asking a follow-up
  for more detail — either add the requested detail or say you don't have more.
- Be concise: 1-4 sentences unless the solution has multiple distinct steps or the user
  asked for an example.
If earlier conversation is shown, keep your answer consistent with it (e.g. the user may
be asking a follow-up).
{history_block}
User's latest message: {question}
Verified problem: {problem}
Verified solution: {solution}

Answer (content only, no greeting or sign-off):"""


def _history_block(state: AgentState) -> str:
    if not state.history:
        return ""
    lines = "\n".join(f"- User: {t.question}\n  Assistant: {t.answer[:200]}" for t in state.history)
    return f"\nEarlier conversation:\n{lines}\n"


def _template_answer(top: ScoredKBEntry) -> str:
    return (
        f"Vấn đề tương tự đã gặp: {top.entry.problem}\n"
        f"→ Giải pháp: {top.entry.solution}\n"
    )


def answer(state: AgentState, llm: LLMClient | None) -> AgentState:
    if not state.retrieved:
        state.answer = NO_ANSWER
        return state

    top = state.retrieved[0]

    if llm is not None and not isinstance(llm, NullLLM):
        prompt = POLISH_PROMPT.format(
            history_block=_history_block(state),
            question=state.question,
            problem=top.entry.problem,
            solution=top.entry.solution,
        )
        try:
            polished = llm.complete(prompt).strip()
            state.answer = f"{polished}\n(nguồn: {top.entry.source_thread_id})"
            return state
        except RuntimeError:
            pass  # LLM unavailable -- fall through to template, never a hard dependency

    state.answer = _template_answer(top)
    return state
