"""Unit tests for the agent nodes and graph routing (no embeddings — uses FakeStore)."""
from __future__ import annotations

import json

from tests.conftest import FakeLLM, FakeStore, make_scored

from src.agents.answerer import NO_ANSWER, answer, direct_reply
from src.agents.critic import critique
from src.agents.orchestrator import orchestrate
from src.agents.retrieval import MIN_RETRIEVAL_SCORE, retrieve
from src.agents.state import AgentState
from src.config import GateThresholds


def _classify_reply(turn_type: str, search_query: str | None = None) -> str:
    return json.dumps({"turn_type": turn_type, "search_query": search_query})


def test_orchestrate_trims_question_and_sets_turn_type():
    llm = FakeLLM(reply=_classify_reply("new_problem", "workflow lỗi"))
    state = orchestrate(AgentState(question="  workflow lỗi  "), llm)
    assert state.question == "workflow lỗi"
    assert state.turn_type == "new_problem"
    assert state.search_query == "workflow lỗi"


def test_orchestrate_chit_chat():
    llm = FakeLLM(reply=_classify_reply("chit_chat"))
    state = orchestrate(AgentState(question="hi"), llm)
    assert state.turn_type == "chit_chat"


def test_orchestrate_off_topic():
    llm = FakeLLM(reply=_classify_reply("off_topic"))
    state = orchestrate(AgentState(question="how do I cook beef pho"), llm)
    assert state.turn_type == "off_topic"


def test_orchestrate_follow_up_rewrites_query():
    llm = FakeLLM(reply=_classify_reply("follow_up", "nguyên nhân workflow mất publish là gì"))
    from src.agents.state import Turn

    state = orchestrate(
        AgentState(question="nguyên nhân là gì?", history=[Turn(question="workflow mất publish", answer="...")]),
        llm,
    )
    assert state.turn_type == "follow_up"
    assert state.search_query == "nguyên nhân workflow mất publish là gì"


def test_orchestrate_falls_back_to_new_problem_on_unparseable_reply():
    # a malformed/garbage LLM reply must not silently misroute or drop the question --
    # fall back to the safest existing behavior (treat as new_problem, search on raw text).
    llm = FakeLLM(reply="not json at all")
    state = orchestrate(AgentState(question="workflow lỗi"), llm)
    assert state.turn_type == "new_problem"
    assert state.search_query == "workflow lỗi"


def test_retrieve_keeps_above_threshold():
    store = FakeStore([make_scored("p", "s", "credential", 0.8)])
    state = retrieve(AgentState(question="q", search_query="q"), store, FakeLLM(reply="yes"), top_k=3)
    assert len(state.retrieved) == 1


def test_retrieve_drops_below_threshold():
    low = MIN_RETRIEVAL_SCORE - 0.1
    store = FakeStore([make_scored("p", "s", "credential", low)])
    state = retrieve(AgentState(question="q", search_query="q"), store, FakeLLM(reply="yes"), top_k=3)
    assert state.retrieved == []


def test_retrieve_strong_score_survives_llm_no():
    # A high-similarity hit (>= LLM_GRADE_TRUST_SCORE) must NOT be vetoed even if a shaky
    # small LLM says "no" -- this is the bug fix for good answers being wrongly rejected.
    from src.agents.retrieval import LLM_GRADE_TRUST_SCORE

    store = FakeStore([make_scored("p", "s", "workflow_publish", LLM_GRADE_TRUST_SCORE + 0.1)])
    llm = FakeLLM(reply="no")
    state = retrieve(AgentState(question="q", search_query="q"), store, llm, top_k=3)
    assert len(state.retrieved) == 1
    assert llm.calls == []  # not even asked -- score alone is trusted


def test_retrieve_middle_band_vetoed_by_llm_no():
    # A borderline hit (between MIN and TRUST) IS subject to the LLM veto.
    from src.agents.retrieval import LLM_GRADE_TRUST_SCORE

    mid = (MIN_RETRIEVAL_SCORE + LLM_GRADE_TRUST_SCORE) / 2
    store = FakeStore([make_scored("p", "s", "connection_access", mid)])
    state = retrieve(AgentState(question="q", search_query="q"), store, FakeLLM(reply="no"), top_k=3)
    assert state.retrieved == []


def test_retrieve_middle_band_kept_on_llm_yes():
    from src.agents.retrieval import LLM_GRADE_TRUST_SCORE

    mid = (MIN_RETRIEVAL_SCORE + LLM_GRADE_TRUST_SCORE) / 2
    store = FakeStore([make_scored("p", "s", "connection_access", mid)])
    state = retrieve(AgentState(question="q", search_query="q"), store, FakeLLM(reply="yes"), top_k=3)
    assert len(state.retrieved) == 1


def test_answer_no_results_returns_apology():
    out = answer(AgentState(question="q", retrieved=[]), FakeLLM(reply="placeholder"))
    assert out.answer == NO_ANSWER


def test_answer_llm_synthesis_used_and_cites_source():
    state = AgentState(question="q", retrieved=[make_scored("mất publish", "kiểm tra vmail", "workflow_publish", 0.9)])
    out = answer(state, FakeLLM(reply="Chào bạn, hãy kiểm tra vmail nhé."))
    assert "Chào bạn" in out.answer
    assert "nguồn" in out.answer


def test_answer_prompt_forbids_greeting_and_signoff():
    # regression guard: every answer used to open with "Chào bạn," and close with a
    # sign-off ("Chúc bạn...", "Nếu cần hỗ trợ thêm...") regardless of the question --
    # the prompt must explicitly forbid this, not just ask for "friendly".
    llm = FakeLLM(reply="placeholder")
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "credential", 0.9)])
    answer(state, llm)
    assert len(llm.calls) == 1
    prompt = llm.calls[0]
    assert "NEVER open with" in prompt
    assert "NEVER close with" in prompt


def test_answer_prompt_permits_illustrating_examples():
    # regression guard: a user asking "lấy ví dụ cách làm giúp tôi" (give me an example)
    # used to get the exact same abstract explanation repeated verbatim, because the
    # prompt forbade "inventing details beyond the solution" -- which also blocked any
    # concrete illustration of it. The prompt must allow illustrating the verified
    # solution with one worked instance, while still forbidding new facts/causes/fixes.
    llm = FakeLLM(reply="placeholder")
    state = AgentState(question="lấy ví dụ cách làm giúp tôi", retrieved=[make_scored("p", "s", "llm_model", 0.9)])
    answer(state, llm)
    prompt = llm.calls[0]
    assert "you MAY illustrate the" in prompt
    assert "not just repeat your previous answer word-for-word" in prompt.lower()


def test_direct_reply_sets_decision_and_skips_kb():
    llm = FakeLLM(reply="Chào bạn! Mình có thể hỗ trợ các câu hỏi kỹ thuật về netAgent.")
    state = direct_reply(AgentState(question="hi", turn_type="chit_chat"), llm)
    assert state.decision == "direct_reply"
    assert state.answer == "Chào bạn! Mình có thể hỗ trợ các câu hỏi kỹ thuật về netAgent."


def test_critic_auto_reply_on_strong_match():
    # score 0.9 * confidence 0.95 = 0.855 >= 0.75
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "credential", 0.9, confidence=0.95)])
    out = critique(state, FakeLLM(), GateThresholds())
    assert out.decision == "auto_reply"
    assert out.confidence > 0.75


def test_critic_suggest_to_staff_on_medium_match():
    # score 0.65 * 0.95 = 0.6175 -> between suggest_to_staff_min (0.55) and auto_reply_min
    # (0.75), with real margin on both sides (not sitting right at a boundary)
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "credential", 0.65, confidence=0.95)])
    out = critique(state, FakeLLM(), GateThresholds())
    assert out.decision == "suggest_to_staff"


def test_critic_escalate_below_suggest_floor():
    # score 0.55 * 0.95 = 0.5225 -> below suggest_to_staff_min (0.55). This threshold was
    # raised from 0.45 after a user-reported case: a wrong-topic KB match scored 0.58
    # composite and was shown as an answer instead of escalating -- calibrated against
    # measured composite scores on 12 known-correct queries (0.62-0.76, see RESULTS.md),
    # so 0.55 sits with real margin below genuinely correct matches.
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "credential", 0.55, confidence=0.95)])
    out = critique(state, FakeLLM(), GateThresholds())
    assert out.decision == "escalate"


def test_critic_escalate_on_no_results():
    out = critique(AgentState(question="q", retrieved=[]), FakeLLM(), GateThresholds())
    assert out.decision == "escalate"
    assert out.confidence == 0.0


def test_critic_low_confidence_entry_does_not_auto_reply():
    # strong query match (0.95) but weak entry (0.3): 0.285 -> escalate, not auto_reply
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "other", 0.95, confidence=0.3)])
    out = critique(state, FakeLLM(), GateThresholds())
    assert out.decision == "escalate"


def test_graph_escalate_fires_alerter(empty_store, recording_alerter):
    from src.agents.graph import build_graph, run_graph

    llm = FakeLLM(reply=_classify_reply("new_problem", "how do I cook beef pho"))
    graph = build_graph(empty_store, llm, recording_alerter)
    state = run_graph(graph, "how do I cook beef pho")
    assert state.decision == "escalate"
    assert len(recording_alerter.sent) == 1
    assert recording_alerter.sent[0].severity == "escalate"


def test_graph_auto_reply_does_not_fire_alerter(high_score_store, recording_alerter):
    from src.agents.graph import build_graph, run_graph

    class ScriptedLLM:
        """Returns a different canned reply per call: classify -> grade -> answer."""

        def __init__(self, replies: list[str]):
            self.replies = list(replies)
            self.calls: list[str] = []

        def complete(self, prompt: str) -> str:
            self.calls.append(prompt)
            return self.replies.pop(0) if self.replies else "yes"

    llm = ScriptedLLM(
        [
            _classify_reply("new_problem", "workflow mất publish"),  # orchestrate
            "kiểm tra credential vmail nhé",  # answer polish
        ]
    )
    graph = build_graph(high_score_store, llm, recording_alerter)
    state = run_graph(graph, "workflow mất publish")
    assert state.decision in ("auto_reply", "suggest_to_staff")
    assert recording_alerter.sent == []


def test_graph_chit_chat_skips_kb_and_escalation(empty_store, recording_alerter):
    from src.agents.graph import build_graph, run_graph

    class ChitChatLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return _classify_reply("chit_chat")
            return "Chào bạn! Mình hỗ trợ các câu hỏi kỹ thuật về netAgent."

    graph = build_graph(empty_store, ChitChatLLM(), recording_alerter)
    state = run_graph(graph, "hi")
    assert state.turn_type == "chit_chat"
    assert state.decision == "direct_reply"
    assert state.retrieved == []
    assert recording_alerter.sent == []
