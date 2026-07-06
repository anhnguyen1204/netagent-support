"""Unit tests for src/pipeline/classify.py — system detection + rule-based labels."""
from __future__ import annotations

import pandas as pd

from src.pipeline.classify import (
    INTENTS,
    SENDER_TYPES,
    TOPICS,
    _rule_based_classify,
    classify,
    is_system_noise,
)
from src.llm.null_llm import NullLLM


def test_is_system_noise_join_leave():
    assert is_system_noise("dunglt36 tham gia nhóm.")
    assert is_system_noise("@dunglt36 Rời nhóm.")
    assert is_system_noise("tinhvv2 được thêm vào nhóm bởi cuongln3.")


def test_is_system_noise_pin():
    assert is_system_noise("@huydd đã ghim một tin nhắn.")
    assert is_system_noise("@duonglt18 pinned a message.")


def test_is_system_noise_false_for_real_message():
    assert not is_system_noise("workflow của em bị mất publish")


def test_rule_based_system_message():
    sender, intent, topic, conf = _rule_based_classify("dunglt36 tham gia nhóm.")
    assert sender == "system"
    assert intent == "none"
    assert topic == "none"


def test_rule_based_returns_valid_taxonomy_values():
    # every output must be a value from the taxonomy so eval/threading stay consistent
    for msg in [
        "workflow bị lỗi credential",
        "cho em hỏi có node vẽ chart không ạ",
        "Do hạ tầng bên e lỗi, bên e đang fix r a nhé",
        "@thangnt30",
    ]:
        sender, intent, topic, conf = _rule_based_classify(msg)
        assert sender in SENDER_TYPES
        assert intent in INTENTS
        assert topic in TOPICS
        assert 0.0 <= conf <= 1.0


def test_rule_based_credential_topic():
    _, _, topic, _ = _rule_based_classify("anh vừa chạy bị lỗi credential")
    assert topic == "credential"


def test_classify_dataframe_adds_columns(tmp_path):
    df = pd.DataFrame(
        {
            "content": ["dunglt36 tham gia nhóm.", "workflow bị lỗi credential nhờ giúp"],
            "user_id": ["u1", "u2"],
            "created_at": [1000.0, 2000.0],
        }
    )
    cache = tmp_path / "cache.json"
    out = classify(df, NullLLM(), cache)
    assert list(out["sender_type"]) == ["system", "customer"]
    assert {"sender_type", "intent", "topic", "confidence"} <= set(out.columns)
    assert cache.exists()  # cache written


def test_classify_uses_cache(tmp_path):
    df = pd.DataFrame({"content": ["workflow bị lỗi"], "user_id": ["u1"], "created_at": [1.0]})
    cache = tmp_path / "cache.json"
    classify(df, NullLLM(), cache)
    mtime = cache.stat().st_mtime_ns
    # second run should hit cache and not rewrite the file
    classify(df, NullLLM(), cache)
    assert cache.stat().st_mtime_ns == mtime
