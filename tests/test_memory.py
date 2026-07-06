"""Unit tests for conversation memory (ConversationStore) and history-aware agents."""
from __future__ import annotations

from tests.conftest import FakeLLM, FakeStore, make_scored

from src.agents.answerer import answer
from src.agents.memory import ConversationStore
from src.agents.retrieval import retrieve
from src.agents.state import AgentState, Turn


def test_store_no_session_returns_empty():
    store = ConversationStore()
    assert store.history(None) == []
    assert store.history("never-seen") == []


def test_store_append_and_read_back():
    store = ConversationStore()
    store.append("s1", "câu hỏi 1", "trả lời 1")
    store.append("s1", "câu hỏi 2", "trả lời 2")
    hist = store.history("s1")
    assert [t.question for t in hist] == ["câu hỏi 1", "câu hỏi 2"]


def test_store_sessions_are_isolated():
    store = ConversationStore()
    store.append("a", "qa", "aa")
    store.append("b", "qb", "ab")
    assert len(store.history("a")) == 1
    assert store.history("a")[0].question == "qa"


def test_store_caps_turns_per_session():
    store = ConversationStore(max_turns=3)
    for i in range(5):
        store.append("s", f"q{i}", f"a{i}")
    hist = store.history("s")
    assert len(hist) == 3
    assert [t.question for t in hist] == ["q2", "q3", "q4"]  # oldest evicted


def test_store_none_session_id_is_noop():
    store = ConversationStore()
    store.append(None, "q", "a")  # should not raise or store anything
    assert store.history(None) == []


def test_retrieve_rewrites_followup_with_history_and_llm():
    # FakeLLM returns a fixed rewritten query; retrieval should search with it and record
    # it on state.search_query.
    store = FakeStore([make_scored("p", "s", "credential", 0.9)])
    llm = FakeLLM(reply="lỗi credential token hết hạn")
    state = AgentState(
        question="còn cái đó thì sao?",
        history=[Turn(question="workflow lỗi", answer="kiểm tra credential")],
    )
    out = retrieve(state, store, llm, top_k=3)
    assert out.search_query == "lỗi credential token hết hạn"


def test_retrieve_no_history_uses_question_verbatim():
    store = FakeStore([make_scored("p", "s", "credential", 0.9)])
    llm = FakeLLM(reply="SHOULD-NOT-BE-USED")
    state = AgentState(question="lỗi credential", history=[])
    out = retrieve(state, store, llm, top_k=3)
    assert out.search_query == "lỗi credential"  # no rewrite without history


def test_answer_includes_history_in_llm_prompt():
    llm = FakeLLM(reply="Trả lời có ngữ cảnh")
    state = AgentState(
        question="còn nó thì sao",
        history=[Turn(question="workflow mất publish", answer="kiểm tra vmail")],
        retrieved=[make_scored("p", "s", "workflow_publish", 0.9)],
    )
    answer(state, llm)
    # the earlier turn's text must appear in the prompt the LLM received
    assert any("workflow mất publish" in prompt for prompt in llm.calls)
