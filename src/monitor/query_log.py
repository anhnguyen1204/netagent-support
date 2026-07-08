"""Per-question JSONL query logging.

Every /ask call is appended as one JSON line: question, rewritten search query,
retrieved candidates + scores, composite confidence, decision, latency. Turns every
live/manual test into a durable, replayable data point instead of a one-off screenshot,
and is the raw material for growing data/qa_golden.csv from real usage.

Disabled (no-op) when constructed with path=None, so it's a one-line toggle, not a
branch scattered through the handler.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.agents.state import AgentState


class QueryLogger:
    def __init__(self, path: Path | None):
        self.path = path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, state: AgentState, session_id: str | None, latency_ms: float, llm_backend: str) -> None:
        if self.path is None:
            return
        record = {
            "ts": time.time(),
            "session_id": session_id,
            "question": state.question,
            "search_query": state.search_query,
            "retrieved": [
                {"source": r.entry.source_thread_id, "problem": r.entry.problem, "score": r.score}
                for r in state.retrieved
            ],
            "composite_confidence": state.confidence,
            "decision": state.decision,
            "latency_ms": latency_ms,
            "llm_backend": llm_backend,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
