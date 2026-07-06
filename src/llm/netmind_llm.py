"""LLMClient stub for the company netMind gateway.

Fill in if/when access is granted. Until then this raises NotImplementedError
unconditionally — LLM_BACKEND=netmind is not a supported runtime mode yet.
"""
from __future__ import annotations

from src.llm.base import LLMClient


class NetMindLLM(LLMClient):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def complete(self, prompt: str) -> str:
        raise NotImplementedError