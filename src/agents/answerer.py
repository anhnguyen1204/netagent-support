"""B3: compose grounded answer from the top retrieved KB entry, via LLM polish."""
from __future__ import annotations

from src.agents.state import AgentState
from src.llm.base import LLMClient

NO_ANSWER = "Xin lỗi, hiện chưa tìm thấy giải pháp phù hợp trong cơ sở tri thức cho câu hỏi này."

DIRECT_REPLY_PROMPT = """You are a Vietnamese-speaking assistant for the netAgent support
chat, replying directly inside the chat UI (no separate human agent wraps your reply).

The user's message is small talk or unrelated to netAgent/netFlow support (turn_type=
{turn_type}). Reply naturally and briefly, matching the message's actual intent:
- Greeting ("hi", "chào bạn"): greet back briefly and mention you can help with
  netAgent/netFlow technical questions.
- Thanks/acknowledgment ("cảm ơn", "ok"): a short, warm "không có gì" — do NOT treat
  this as a joke or non-sequitur, and do not repeat the "I can help with X" pitch (it's
  redundant after a thank-you).
- Meta question about you ("bạn là ai"): briefly say you are a netAgent/netFlow support
  assistant. Do NOT invent any other detail about yourself (company, vendor, model,
  version) that isn't stated here.
- Off-topic (unrelated to netAgent/netFlow, e.g. weather/food/general knowledge):
  politely say you can only help with netAgent/netFlow support questions.
Hard rules: reply ONLY in Vietnamese, never mixing in English or any other language.
Keep it to 1-2 sentences. No greetings-within-greetings, no corporate sign-off filler.

User's message: {question}

Reply:"""

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


def answer(state: AgentState, llm: LLMClient) -> AgentState:
    if not state.retrieved:
        state.answer = NO_ANSWER
        return state

    top = state.retrieved[0]
    prompt = POLISH_PROMPT.format(
        history_block=_history_block(state),
        question=state.question,
        problem=top.entry.problem,
        solution=top.entry.solution,
    )
    polished = llm.complete(prompt).strip()
    state.answer = f"{polished}\n(nguồn: {top.entry.source_thread_id})"
    return state


def direct_reply(state: AgentState, llm: LLMClient) -> AgentState:
    """chit_chat/off_topic path: no KB involved, just a short natural reply."""
    prompt = DIRECT_REPLY_PROMPT.format(turn_type=state.turn_type, question=state.question)
    state.answer = llm.complete(prompt).strip()
    state.decision = "direct_reply"
    return state
