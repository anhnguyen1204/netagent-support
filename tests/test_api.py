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


def test_ask_irrelevant_escalates(client):
    r = client.post("/ask", json={"question": "how do I cook beef pho"})
    body = r.json()
    assert body["decision"] == "escalate"
    assert body["source_thread_id"] is None


def test_ask_records_topic_for_spike_monitor(client):
    client.post("/ask", json={"question": "workflow mất publish liên tục"})
    monitor = app.state.spike_monitor
    total = sum(sum(buckets.values()) for buckets in monitor._counts.values())
    assert total >= 1
