"""KB entry schema: problem, solution, topic, source, confidence."""
from __future__ import annotations

from pydantic import BaseModel


class KBEntry(BaseModel):
    problem: str
    solution: str
    topic: str
    source_thread_id: str
    confidence: float
    created_at: float  # unix epoch ms


class ScoredKBEntry(BaseModel):
    entry: KBEntry
    score: float