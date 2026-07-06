"""Template-only fallback LLMClient — zero LLM calls.

Used when LLM_BACKEND=null. classify.py falls back to keyword rules for sender_type
only when this backend is active; answerer.py falls back to template-based answers.
"""
from __future__ import annotations

from src.llm.base import LLMClient


class NullLLM(LLMClient):
    def complete(self, prompt: str) -> str:
        raise RuntimeError(
            "NullLLM has no LLM backend configured — this is expected. Callers must "
            "check for NullLLM and use their rule-based/template fallback instead of "
            "calling complete()."
        )