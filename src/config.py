"""Central configuration. Gate thresholds and runtime settings live here, not scattered
across modules.
"""
from __future__ import annotations

import os

from pydantic import BaseModel


class GateThresholds(BaseModel):
    """Confidence thresholds that decide the agent's final action (src/agents/critic.py).

    confidence >= auto_reply_min          -> auto_reply
    confidence >= suggest_to_staff_min    -> suggest_to_staff
    below suggest_to_staff_min            -> escalate
    """

    auto_reply_min: float = 0.75
    suggest_to_staff_min: float = 0.45


class Settings(BaseModel):
    llm_backend: str
    ollama_host: str
    ollama_model: str
    netmind_api_url: str
    netmind_api_key: str

    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str

    embedding_model: str
    embedding_model_fallback: str

    alerter_backend: str

    host: str
    port: int

    gate: GateThresholds


def load_settings() -> Settings:
    return Settings(
        llm_backend=os.getenv("LLM_BACKEND", "null"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        netmind_api_url=os.getenv("NETMIND_API_URL", ""),
        netmind_api_key=os.getenv("NETMIND_API_KEY", ""),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "netagent_kb"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_model_fallback=os.getenv(
            "EMBEDDING_MODEL_FALLBACK", "paraphrase-multilingual-mpnet-base-v2"
        ),
        alerter_backend=os.getenv("ALERTER_BACKEND", "console"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        gate=GateThresholds(),
    )
