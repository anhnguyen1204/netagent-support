"""CLI: start the FastAPI server.

All runtime config (LLM backend, Qdrant, embedding model, gate thresholds) is read from
the environment by `load_settings()` inside `src.server` at app startup — see
`.env.example`. This launcher only needs the host/port to bind uvicorn.

Examples:
    python scripts/run_server.py                       # LLM_BACKEND defaults to null
    LLM_BACKEND=ollama OLLAMA_MODEL=qwen2.5:3b python scripts/run_server.py
"""
from __future__ import annotations

import uvicorn

from src.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run("src.server:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
