"""Bước 1: sender/intent/topic labeling.

Regex pass for system/noise first, then for the rest: NullLLM uses a rule-based
classifier (keyword/regex signals covering sender_type, intent, and topic — the
BUILD_PLAN spec calls for sender_type-only on NullLLM, but a same-cost rule pass over
intent/topic gives classify.py a non-trivial baseline instead of "unknown" everywhere,
which real LLM backends are then compared against). Any other LLMClient gets a single
few-shot prompt per message requesting {sender_type, intent, topics[], confidence} as
JSON. Caches responses by content hash in data/processed/classify_cache.json.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.llm.base import LLMClient
from src.llm.null_llm import NullLLM

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

STAFF_RE = re.compile(
    r"Team netAgent|bọn em (sẽ|đang)|em (sẽ|đang) (note|kiểm tra|xử lý|báo)|"
    r"để em|để bọn em|hiện tại (hệ thống|chưa|đã)|"
    r"📢|\[DONE\]|thông báo (lại|rằng)|dự kiến|em note lại|đã fix|sẽ fix|"
    r"e note|a note|do (lỗi|hạ tầng|nguyên nhân)|nguyên nhân (là|do)|khắc phục|"
    r"anh\/chị có thể|dùng node .+ nhé|sử dụng node .+ nhé",
    re.IGNORECASE,
)
CUSTOMER_RE = re.compile(
    r"giúp (em|anh|mình|e|a|c|chị)|nhờ (anh|chị|team|e|a)|xem giúp|check giúp|"
    r"bị lỗi|báo lỗi|ko chạy|không chạy|mất publish|unpublish|vẫn (bị|chưa|lỗi)|"
    r"cho (em|e|anh|a|mình) (hỏi|xin)|làm sao|có thể.*không ạ|được không ạ|"
    r"xin (kết nối|link|api key|quyền)|add giúp|hỗ trợ (em|giúp|với)",
    re.IGNORECASE,
)

REQUEST_ACCESS_RE = re.compile(r"xin (kết nối|link|api key|quyền)|add giúp|mở kết nối", re.IGNORECASE)
REPORT_PROBLEM_RE = re.compile(
    r"bị lỗi|báo lỗi|ko chạy|không chạy|mất publish|unpublish|vẫn (bị|chưa|lỗi)|"
    r"không (vào|dùng) được|timeout|fail", re.IGNORECASE,
)
ASK_QUESTION_RE = re.compile(r"\?|không ạ$|có thể|làm sao|sao .* nhỉ|được không", re.IGNORECASE)
ACKNOWLEDGE_RE = re.compile(r"cảm ơn|cám ơn|Dạ vâng|ok ạ|thank|vâng ạ|got it", re.IGNORECASE)
STATUS_UPDATE_RE = re.compile(r"📢|\[DONE\]|đang (tiến hành |)kiểm tra|sẽ (fix|thông báo)|dự kiến", re.IGNORECASE)
PROVIDE_SOLUTION_RE = re.compile(
    r"dùng node|sử dụng node|nhé (anh|chị|em|a|c)$|thử lại|check lại|xem lại cấu hình|"
    r"do (lỗi|node|hạ tầng)", re.IGNORECASE,
)

TOPIC_RULES = [
    ("workflow_publish", re.compile(r"publish|unpublish|public", re.IGNORECASE)),
    ("credential", re.compile(r"credential|authen|token|mật khẩu|password", re.IGNORECASE)),
    ("datatable", re.compile(r"datatable|data table|import csv|xlsx|storage used|dung lượng", re.IGNORECASE)),
    ("email", re.compile(r"\bemail\b|\bmail\b|imap|vmail|bộ lọc", re.IGNORECASE)),
    ("llm_model", re.compile(r"\bmodel\b|\bLLM\b|gateway|Minimax|Qwen|GPT|DeepSeek|quá tải|netmind", re.IGNORECASE)),
    ("node_feature", re.compile(r"\bnode\b|AI Agent|OCR|Tableau|NetChat|\bCode\b", re.IGNORECASE)),
    ("connection_access", re.compile(r"N8N|kết nối|database|đăng nhập|truy cập|ITBrain|IP máy", re.IGNORECASE)),
    ("infra_incident", re.compile(r"K8s|hạ tầng|sự cố|outage|DGX", re.IGNORECASE)),
    ("workflow_run", re.compile(r"chạy|run|flow .* (lần|tay)|tự động", re.IGNORECASE)),
]


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


def _rule_based_classify(content: str) -> tuple[str, str, str, float]:
    """Rule-based fallback used by NullLLM. Returns (sender_type, intent, topic, confidence)."""
    c = str(content)

    if is_system_noise(c):
        return "system", "none", "none", 0.95

    staff_hit = bool(STAFF_RE.search(c))
    customer_hit = bool(CUSTOMER_RE.search(c))
    if staff_hit and not customer_hit:
        sender_type, sender_conf = "staff", 0.6
    elif customer_hit and not staff_hit:
        sender_type, sender_conf = "customer", 0.6
    elif staff_hit and customer_hit:
        sender_type, sender_conf = "staff", 0.4
    else:
        sender_type, sender_conf = "unknown", 0.0

    if sender_type == "unknown":
        intent = "none"
    elif STATUS_UPDATE_RE.search(c):
        intent = "status_update"
    elif sender_type == "staff" and PROVIDE_SOLUTION_RE.search(c):
        intent = "provide_solution"
    elif REQUEST_ACCESS_RE.search(c):
        intent = "request_access"
    elif REPORT_PROBLEM_RE.search(c):
        intent = "report_problem"
    elif ACKNOWLEDGE_RE.search(c):
        intent = "acknowledge"
    elif ASK_QUESTION_RE.search(c):
        intent = "ask_question"
    elif sender_type == "staff":
        intent = "provide_solution"
    else:
        intent = "ask_question"

    if sender_type == "unknown":
        topic = "none"
    else:
        topic = "other"
        for name, rx in TOPIC_RULES:
            if rx.search(c):
                topic = name
                break

    return sender_type, intent, topic, sender_conf


def _llm_few_shot_classify(content: str, llm: LLMClient) -> tuple[str, str, str, float]:
    """LLM few-shot classification for any real (non-Null) LLMClient backend."""
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
    use_rules = isinstance(llm, NullLLM)

    sender_types, intents, topics, confidences = [], [], [], []
    cache_dirty = False

    for content in df["content"].astype(str):
        key = _content_hash(content)
        if key in cache:
            sender_type, intent, topic, confidence = cache[key]
        else:
            if use_rules:
                sender_type, intent, topic, confidence = _rule_based_classify(content)
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