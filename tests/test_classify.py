"""Unit tests for src/pipeline/classify.py — system detection + LLM classification."""
from __future__ import annotations

import pandas as pd

from tests.conftest import FakeLLM

from src.pipeline.classify import (
    INTENTS,
    SENDER_TYPES,
    TOPICS,
    classify,
    is_system_noise,
)


def test_is_system_noise_join_leave():
    assert is_system_noise("dunglt36 tham gia nhóm.")
    assert is_system_noise("@dunglt36 Rời nhóm.")
    assert is_system_noise("tinhvv2 được thêm vào nhóm bởi cuongln3.")


def test_is_system_noise_pin():
    assert is_system_noise("@huydd đã ghim một tin nhắn.")
    assert is_system_noise("@duonglt18 pinned a message.")


def test_is_system_noise_false_for_real_message():
    assert not is_system_noise("workflow của em bị mất publish")


def test_classify_system_message_short_circuits_without_llm_call(tmp_path):
    # system-noise messages never need an LLM call -- deterministic label instead.
    df = pd.DataFrame({"content": ["dunglt36 tham gia nhóm."], "user_id": ["u1"], "created_at": [1.0]})
    cache = tmp_path / "cache.json"
    llm = FakeLLM(reply="should never be called")
    out = classify(df, llm, cache)
    assert list(out["sender_type"]) == ["system"]
    assert list(out["intent"]) == ["none"]
    assert list(out["topic"]) == ["none"]
    assert llm.calls == []


def test_classify_dataframe_adds_columns(tmp_path):
    df = pd.DataFrame(
        {
            "content": ["workflow bị lỗi credential nhờ giúp"],
            "user_id": ["u2"],
            "created_at": [2000.0],
        }
    )
    cache = tmp_path / "cache.json"
    llm = FakeLLM(
        reply='{"sender_type": "customer", "intent": "report_problem", "topic": "credential", "confidence": 0.8}'
    )
    out = classify(df, llm, cache)
    assert list(out["sender_type"]) == ["customer"]
    assert list(out["topic"]) == ["credential"]
    assert {"sender_type", "intent", "topic", "confidence"} <= set(out.columns)
    assert cache.exists()  # cache written


def test_classify_result_values_are_valid_taxonomy(tmp_path):
    df = pd.DataFrame({"content": ["cho em hỏi có node vẽ chart không ạ"], "user_id": ["u1"], "created_at": [1.0]})
    cache = tmp_path / "cache.json"
    llm = FakeLLM(
        reply='{"sender_type": "customer", "intent": "ask_question", "topic": "node_feature", "confidence": 0.7}'
    )
    out = classify(df, llm, cache)
    row = out.iloc[0]
    assert row["sender_type"] in SENDER_TYPES
    assert row["intent"] in INTENTS
    assert row["topic"] in TOPICS
    assert 0.0 <= row["confidence"] <= 1.0


def test_classify_uses_cache(tmp_path):
    df = pd.DataFrame({"content": ["workflow bị lỗi"], "user_id": ["u1"], "created_at": [1.0]})
    cache = tmp_path / "cache.json"
    llm = FakeLLM(
        reply='{"sender_type": "customer", "intent": "report_problem", "topic": "workflow_run", "confidence": 0.6}'
    )
    classify(df, llm, cache)
    mtime = cache.stat().st_mtime_ns
    calls_after_first = len(llm.calls)

    # second run should hit cache: no new LLM call, cache file untouched
    classify(df, llm, cache)
    assert cache.stat().st_mtime_ns == mtime
    assert len(llm.calls) == calls_after_first
