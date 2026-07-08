"""Bước 1: sender/intent/topic labeling.

`SYSTEM_RE` catches join/leave/pin noise deterministically (exact string patterns, no
LLM needed — reused independently by threading/spike code via `is_system_noise`). Every
other message gets a single LLM few-shot prompt requesting {sender_type, intent, topic,
confidence} as JSON. Caches responses by content hash in
data/processed/classify_cache.json.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.llm.base import LLMClient

SENDER_TYPES = ["system", "customer", "staff", "unknown"]
INTENTS = [
    "report_problem", "ask_question", "request_access", "provide_solution",
    "acknowledge", "status_update", "none",
]
TOPICS = [
    "workflow_publish", "workflow_run", "credential", "node_feature", "email",
    "datatable", "llm_model", "connection_access", "infra_incident", "other", "none",
]

SYSTEM_RE = re.compile(
    r"tham gia nhóm|Rời nhóm|được thêm vào nhóm bởi|đã ghim|đã bỏ ghim|bỏ ghim|"
    r"pinned a message|unpinned a message|^\s*\[?https?://\S+/join/|cập nhật tên hiển thị",
    re.IGNORECASE,
)


class ClassifiedMessage(BaseModel):
    user_id: str
    content: str
    created_at: float
    sender_type: str
    intent: str
    topics: list[str]
    confidence: float


def is_system_noise(content: str) -> bool:
    return bool(SYSTEM_RE.search(str(content)))


def _llm_few_shot_classify(content: str, llm: LLMClient) -> tuple[str, str, str, float]:
    """LLM few-shot classification. System-noise messages are still short-circuited to
    the deterministic system/none/none/0.95 label (join/leave/pin text carries no
    ambiguity worth spending an LLM call on)."""
    if is_system_noise(content):
        return "system", "none", "none", 0.95

    prompt = f"""Classify this Vietnamese support-chat message. Respond with ONLY a JSON object.

sender_type: one of {SENDER_TYPES}
intent: one of {INTENTS}
topic: one of {TOPICS}
confidence: float 0-1

Examples:
Message: "dunglt36 tham gia nhóm."
{{"sender_type": "system", "intent": "none", "topic": "none", "confidence": 0.95}}

Message: "anh @thangnt30 xem giúp em flow hôm nay không chạy ạ"
{{"sender_type": "customer", "intent": "report_problem", "topic": "workflow_run", "confidence": 0.85}}

Message: "Do hạ tầng bên e lỗi, bên e đang fix r a nhé"
{{"sender_type": "staff", "intent": "status_update", "topic": "infra_incident", "confidence": 0.8}}

Message: "{content}"
"""
    raw = llm.complete(prompt)
    data = json.loads(raw)
    return (
        data.get("sender_type", "unknown"),
        data.get("intent", "none"),
        data.get("topic", "none"),
        float(data.get("confidence", 0.0)),
    )


def _content_hash(content: str) -> str:
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def classify(df: pd.DataFrame, llm: LLMClient, cache_path: Path) -> pd.DataFrame:
    cache = _load_cache(cache_path)

    sender_types, intents, topics, confidences = [], [], [], []
    cache_dirty = False

    for content in df["content"].astype(str):
        key = _content_hash(content)
        if key in cache:
            sender_type, intent, topic, confidence = cache[key]
        else:
            sender_type, intent, topic, confidence = _llm_few_shot_classify(content, llm)
            cache[key] = [sender_type, intent, topic, confidence]
            cache_dirty = True

        sender_types.append(sender_type)
        intents.append(intent)
        topics.append(topic)
        confidences.append(confidence)

    if cache_dirty:
        _save_cache(cache_path, cache)

    result = df.copy()
    result["sender_type"] = sender_types
    result["intent"] = intents
    result["topic"] = topics
    result["confidence"] = confidences
    return result