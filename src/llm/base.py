"""LLMClient interface — the LLM seam.

Stub today: local Ollama model, or NullLLM (template-only, zero LLM calls).
Real version later: netMind gateway.
Every LLM call in the codebase must go through this interface — never call an API
client directly from agent/pipeline code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract text-completion client."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return a completion for the given prompt. Must never be a hard dependency
        for the system to return an answer — callers need a non-LLM fallback path.
        """
        raise NotImplementedError