"""Integration tests: real embedding model + curated KB in an in-memory Qdrant.

Marked slow (loads the embedding model). Run with `pytest -m slow` or just `pytest`.
Skip with `pytest -m "not slow"`.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_curated_kb_loads(real_kb_store):
    # a handful of diverse queries should each return *some* result
    for q in ["workflow mất publish", "credential lỗi", "datatable đầy"]:
        results = real_kb_store.search(q, top_k=3)
        assert len(results) >= 1


def test_search_ranks_relevant_topic_first(real_kb_store):
    cases = [
        ("workflow của em bị mất publish", "workflow_publish"),
        ("import csv vào datatable bị lỗi header", "datatable"),
        ("không vào được netflow bị 502", "connection_access"),
        ("model Qwen không xử lý ảnh", "llm_model"),
    ]
    for query, expected_topic in cases:
        top = real_kb_store.search(query, top_k=1)[0]
        assert top.entry.topic == expected_topic, f"{query!r} -> got {top.entry.topic}"
        assert top.score > 0.5


def test_search_scores_are_descending(real_kb_store):
    results = real_kb_store.search("workflow lỗi", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
