"""Integration test: the full FastAPI /ask endpoint via TestClient.

Marked slow (starts the app, which loads the embedding model + curated KB + spike
monitor scheduler). Exercises the whole Regime-B path end to end.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.server import app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    # lifespan runs on context enter/exit: builds store+KB+graph, starts/stops scheduler
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_returns_grounded_answer(client):
    r = client.post("/ask", json={"question": "workflow của em bị mất publish"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in ("auto_reply", "suggest_to_staff")
    assert body["source_thread_id"] == "curated"
    assert body["answer"]  # non-empty
    assert body["confidence"] > 0.0


def test_ask_off_topic_gets_direct_reply_not_escalation(client):
    # off_topic/chit_chat turns are classified before retrieval and short-circuit to a
    # direct_reply (no KB search, no escalation/alert) -- see RESULTS.md's "LLM made
    # mandatory + turn-type routing" entry. This used to be an escalate case with a
    # generic "0% confidence" template; it's now a natural redirect reply instead.
    r = client.post("/ask", json={"question": "how do I cook beef pho"})
    body = r.json()
    assert body["decision"] == "direct_reply"
    assert body["source_thread_id"] is None
    assert body["answer"]  # non-empty natural reply, not the old escalation template


def test_ask_genuine_kb_gap_escalates(client):
    # a real technical question with no curated KB entry should still escalate -- this
    # is the confident-wrong-answer failure mode the gate exists to catch.
    r = client.post("/ask", json={"question": "làm sao để reset mật khẩu tài khoản netMind của tôi"})
    body = r.json()
    assert body["decision"] in ("escalate", "suggest_to_staff")


def test_ask_records_topic_for_spike_monitor(client):
    client.post("/ask", json={"question": "workflow mất publish liên tục"})
    monitor = app.state.spike_monitor
    total = sum(sum(buckets.values()) for buckets in monitor._counts.values())
    assert total >= 1
