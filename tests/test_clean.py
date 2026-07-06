"""Unit tests for src/pipeline/clean.py — mention/entity extraction, dedup, timestamps."""
from __future__ import annotations

import pandas as pd

from src.pipeline.clean import (
    dedup,
    extract_entities,
    extract_mentions,
    parse_timestamps,
)


def test_extract_mentions_finds_handles():
    assert extract_mentions("@dongntt @thangnt30 pls help") == ["@dongntt", "@thangnt30"]


def test_extract_mentions_empty_when_none():
    assert extract_mentions("workflow bị lỗi") == []


def test_extract_mentions_handles_non_string():
    assert extract_mentions(None) == []


def test_extract_entities_urls_and_ips():
    text = "check https://netflow.viettel.vn/workflow/abc and 10.230.85.15"
    entities = extract_entities(text)
    assert any("netflow.viettel.vn" in e for e in entities)
    assert "10.230.85.15" in entities


def test_extract_entities_node_tokens():
    assert "bot-netchat" in extract_entities("anh check xem để bot-netchat chưa ạ")


def test_extract_entities_ignores_plain_text():
    assert extract_entities("Dạ vâng em cảm ơn ạ") == []


def test_dedup_drops_exact_triples_only():
    df = pd.DataFrame(
        {
            "userId": ["u1", "u1", "u2"],
            "content": ["same", "same", "same"],
            "create_at": [1000.0, 1000.0, 1000.0],
        }
    )
    # rows 0 and 1 are identical triples -> one dropped; row 2 differs by userId -> kept
    out = dedup(df)
    assert len(out) == 2


def test_dedup_keeps_reposts_at_different_times():
    df = pd.DataFrame(
        {
            "userId": ["u1", "u1"],
            "content": ["same", "same"],
            "create_at": [1000.0, 2000.0],
        }
    )
    assert len(dedup(df)) == 2


def test_parse_timestamps_localizes_to_vietnam():
    df = pd.DataFrame({"create_at": [1780570000000.0]})
    out = parse_timestamps(df)
    # 1780570000000 ms -> 2026-06-04 17:46:40 +07:00
    dt = out["created_at_dt"].iloc[0]
    assert str(dt.tz) == "Asia/Ho_Chi_Minh"
    assert dt.hour == 17
    assert out["created_at"].iloc[0] == 1780570000000.0
