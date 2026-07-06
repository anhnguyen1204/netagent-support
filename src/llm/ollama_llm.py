"""LLMClient implementation backed by a local Ollama server.

Used when LLM_BACKEND=ollama.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from src.llm.base import LLMClient


class OllamaLLM(LLMClient):
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def complete(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} — is `ollama serve` running "
                f"and is model '{self.model}' pulled?"
            ) from exc
        return payload["response"]