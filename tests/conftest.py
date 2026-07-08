"""Shared pytest fixtures.

Two tiers of tests:
- fast unit tests use `fake_store` (no embeddings, no network) and run in ~1s.
- integration tests marked `@pytest.mark.slow` load the real embedding model / build a
  real in-memory Qdrant KB; skip them with `pytest -m "not slow"`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.state import AgentState
from src.alerts.base import AlertRecord, Alerter
from src.kb.schema import KBEntry, ScoredKBEntry

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_KB_CSV = REPO_ROOT / "data" / "curated_kb.csv"


class FakeStore:
    """A stand-in for KBStore that returns a fixed, pre-scored result list without any
    embedding model. Lets agent-node/graph tests run without loading bge-m3.
    """

    def __init__(self, results: list[ScoredKBEntry] | None = None):
        self._results = results or []

    def search(self, query: str, top_k: int) -> list[ScoredKBEntry]:
        return self._results[:top_k]


class RecordingAlerter(Alerter):
    """Captures AlertRecords instead of printing, so tests can assert on them."""

    def __init__(self):
        self.sent: list[AlertRecord] = []

    def send(self, alert: AlertRecord) -> None:
        self.sent.append(alert)


class FakeLLM:
    """A stand-in LLMClient that returns a fixed reply, so LLM-path logic (turn
    classification, relevance grading, answer synthesis) can be tested without a real
    model.
    """

    def __init__(self, reply: str = "yes"):
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.reply


def make_scored(problem: str, solution: str, topic: str, score: float, confidence: float = 0.95) -> ScoredKBEntry:
    return ScoredKBEntry(
        entry=KBEntry(
            problem=problem,
            solution=solution,
            topic=topic,
            source_thread_id="curated",
            confidence=confidence,
            created_at=0.0,
        ),
        score=score,
    )


@pytest.fixture
def recording_alerter() -> RecordingAlerter:
    return RecordingAlerter()


@pytest.fixture
def high_score_store() -> FakeStore:
    return FakeStore([make_scored("workflow mất publish", "kiểm tra credential vmail", "workflow_publish", 0.90)])


@pytest.fixture
def empty_store() -> FakeStore:
    return FakeStore([])


@pytest.fixture(scope="session")
def real_kb_store():
    """A real in-memory Qdrant KBStore loaded with the curated KB. Session-scoped so the
    embedding model loads once. Uses the lighter mpnet model to keep it CPU-friendly.
    """
    from qdrant_client import QdrantClient

    from src.kb.store import KBStore
    from src.pipeline.synthesize_kb import load_curated_kb

    client = QdrantClient(":memory:")
    store = KBStore(
        host="localhost",
        port=6333,
        collection="test_kb",
        embedding_model="paraphrase-multilingual-mpnet-base-v2",
        embedding_model_fallback="paraphrase-multilingual-mpnet-base-v2",
        client=client,
    )
    load_curated_kb(store, CURATED_KB_CSV)
    return store
