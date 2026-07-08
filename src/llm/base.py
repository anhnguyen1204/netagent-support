"""LLMClient interface — the LLM seam.

Stub today: local Ollama model. Real version later: netMind gateway. Every LLM call
in the codebase must go through this interface — never call an API client directly
from agent/pipeline code. LLM access is a hard requirement: there is no rule-based or
template fallback path anywhere in the system.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract text-completion client."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return a completion for the given prompt."""
        raise NotImplementedError