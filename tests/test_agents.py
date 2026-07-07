"""Unit tests for the agent nodes and graph routing (no embeddings — uses FakeStore)."""
from __future__ import annotations

from tests.conftest import FakeStore, make_scored

from src.agents.answerer import NO_ANSWER, answer
from src.agents.critic import critique
from src.agents.orchestrator import orchestrate
from src.agents.retrieval import MIN_RETRIEVAL_SCORE, retrieve
from src.agents.state import AgentState
from src.config import GateThresholds
from src.llm.null_llm import NullLLM


def test_orchestrate_trims_question():
    state = orchestrate(AgentState(question="  workflow lỗi  "))
    assert state.question == "workflow lỗi"


def test_retrieve_keeps_above_threshold():
    store = FakeStore([make_scored("p", "s", "credential", 0.8)])
    state = retrieve(AgentState(question="q"), store, NullLLM(), top_k=3)
    assert len(state.retrieved) == 1


def test_retrieve_drops_below_threshold():
    low = MIN_RETRIEVAL_SCORE - 0.1
    store = FakeStore([make_scored("p", "s", "credential", low)])
    state = retrieve(AgentState(question="q"), store, NullLLM(), top_k=3)
    assert state.retrieved == []


def test_retrieve_strong_score_survives_llm_no():
    # A high-similarity hit (>= LLM_GRADE_TRUST_SCORE) must NOT be vetoed even if a shaky
    # small LLM says "no" -- this is the bug fix for good answers being wrongly rejected.
    from src.agents.retrieval import LLM_GRADE_TRUST_SCORE
    from tests.conftest import FakeLLM

    store = FakeStore([make_scored("p", "s", "workflow_publish", LLM_GRADE_TRUST_SCORE + 0.1)])
    llm = FakeLLM(reply="no")
    state = retrieve(AgentState(question="q"), store, llm, top_k=3)
    assert len(state.retrieved) == 1
    assert llm.calls == []  # not even asked -- score alone is trusted


def test_retrieve_middle_band_vetoed_by_llm_no():
    # A borderline hit (between MIN and TRUST) IS subject to the LLM veto.
    from src.agents.retrieval import LLM_GRADE_TRUST_SCORE
    from tests.conftest import FakeLLM

    mid = (MIN_RETRIEVAL_SCORE + LLM_GRADE_TRUST_SCORE) / 2
    store = FakeStore([make_scored("p", "s", "connection_access", mid)])
    state = retrieve(AgentState(question="q"), store, FakeLLM(reply="no"), top_k=3)
    assert state.retrieved == []


def test_retrieve_middle_band_kept_on_llm_yes():
    from src.agents.retrieval import LLM_GRADE_TRUST_SCORE
    from tests.conftest import FakeLLM

    mid = (MIN_RETRIEVAL_SCORE + LLM_GRADE_TRUST_SCORE) / 2
    store = FakeStore([make_scored("p", "s", "connection_access", mid)])
    state = retrieve(AgentState(question="q"), store, FakeLLM(reply="yes"), top_k=3)
    assert len(state.retrieved) == 1


def test_answer_template_includes_problem_and_solution():
    state = AgentState(question="q", retrieved=[make_scored("mất publish", "kiểm tra vmail", "workflow_publish", 0.9)])
    out = answer(state, NullLLM())
    assert "kiểm tra vmail" in out.answer  # solution
    assert "mất publish" in out.answer  # problem


def test_answer_no_results_returns_apology():
    out = answer(AgentState(question="q", retrieved=[]), NullLLM())
    assert out.answer == NO_ANSWER


def test_answer_llm_synthesis_used_and_cites_source():
    # with a real LLM backend, the answer is the LLM's text (still with a source citation)
    from tests.conftest import FakeLLM

    state = AgentState(question="q", retrieved=[make_scored("mất publish", "kiểm tra vmail", "workflow_publish", 0.9)])
    out = answer(state, FakeLLM(reply="Chào bạn, hãy kiểm tra vmail nhé."))
    assert "Chào bạn" in out.answer
    assert "nguồn" in out.answer


def test_answer_prompt_forbids_greeting_and_signoff():
    # regression guard: every answer used to open with "Chào bạn," and close with a
    # sign-off ("Chúc bạn...", "Nếu cần hỗ trợ thêm...") regardless of the question --
    # the prompt must explicitly forbid this, not just ask for "friendly".
    from tests.conftest import FakeLLM

    llm = FakeLLM(reply="placeholder")
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "credential", 0.9)])
    answer(state, llm)
    assert len(llm.calls) == 1
    prompt = llm.calls[0]
    assert "NEVER open with" in prompt
    assert "NEVER close with" in prompt


def test_critic_auto_reply_on_strong_match():
    # score 0.9 * confidence 0.95 = 0.855 >= 0.75
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "credential", 0.9, confidence=0.95)])
    out = critique(state, NullLLM(), GateThresholds())
    assert out.decision == "auto_reply"
    assert out.confidence > 0.75


def test_critic_suggest_to_staff_on_medium_match():
    # score 0.6 * 0.95 = 0.57 -> between 0.45 and 0.75
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "credential", 0.6, confidence=0.95)])
    out = critique(state, NullLLM(), GateThresholds())
    assert out.decision == "suggest_to_staff"


def test_critic_escalate_on_no_results():
    out = critique(AgentState(question="q", retrieved=[]), NullLLM(), GateThresholds())
    assert out.decision == "escalate"
    assert out.confidence == 0.0


def test_critic_low_confidence_entry_does_not_auto_reply():
    # strong query match (0.95) but weak entry (0.3): 0.285 -> escalate, not auto_reply
    state = AgentState(question="q", retrieved=[make_scored("p", "s", "other", 0.95, confidence=0.3)])
    out = critique(state, NullLLM(), GateThresholds())
    assert out.decision == "escalate"


def test_graph_escalate_fires_alerter(empty_store, recording_alerter):
    from src.agents.graph import build_graph, run_graph

    graph = build_graph(empty_store, NullLLM(), recording_alerter)
    state = run_graph(graph, "how do I cook beef pho")
    assert state.decision == "escalate"
    assert len(recording_alerter.sent) == 1
    assert recording_alerter.sent[0].severity == "escalate"


def test_graph_auto_reply_does_not_fire_alerter(high_score_store, recording_alerter):
    from src.agents.graph import build_graph, run_graph

    graph = build_graph(high_score_store, NullLLM(), recording_alerter)
    state = run_graph(graph, "workflow mất publish")
    assert state.decision in ("auto_reply", "suggest_to_staff")
    assert recording_alerter.sent == []
