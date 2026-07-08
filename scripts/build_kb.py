"""CLI: run the offline Regime A pipeline end-to-end, with progress logging.

Steps: clean -> classify -> load the (curated) Knowledge Base into the vector DB.

The KB is built from the hand-curated `data/curated_kb.csv`, not auto-extracted from
conversation threads — see RESULTS.md Phase 4 for why (this dataset yields almost no
clean resolved problem->solution exchanges; the thread-extraction path was removed).

clean + classify are kept here because they produce `clean_messages.parquet` (consumed
by the spike monitor / replay demo) and the cached classifications.
"""
from __future__ import annotations

import os
from pathlib import Path

from qdrant_client import QdrantClient

from src.kb.store import KBStore
from src.llm.base import LLMClient
from src.llm.ollama_llm import OllamaLLM
from src.pipeline.classify import classify
from src.pipeline.clean import clean
from src.pipeline.synthesize_kb import load_curated_kb

RAW_CSV = Path("data/raw/output.csv")
CLEAN_PARQUET = Path("data/processed/clean_messages.parquet")
CLASSIFY_CACHE = Path("data/processed/classify_cache.json")
CURATED_KB_CSV = Path("data/curated_kb.csv")


def _build_llm(backend: str) -> LLMClient:
    if backend == "ollama":
        return OllamaLLM(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        )
    raise RuntimeError(
        f"Unknown or unconfigured LLM_BACKEND={backend!r} -- LLM access is a hard "
        "requirement now. Set LLM_BACKEND=ollama (and have `ollama serve` running)."
    )


def _build_kb_store() -> KBStore:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    try:
        client = QdrantClient(host=host, port=port)
        client.get_collections()  # cheap call to confirm the server is actually reachable
    except Exception:
        print(f"could not reach Qdrant at {host}:{port} -- using in-memory store for this run")
        client = QdrantClient(":memory:")
    return KBStore(
        host=host,
        port=port,
        collection=os.getenv("QDRANT_COLLECTION", "netagent_kb"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_model_fallback="paraphrase-multilingual-mpnet-base-v2",
        client=client,
    )


def main() -> None:
    print("=== Step 1: clean ===")
    clean_df = clean(RAW_CSV, CLEAN_PARQUET)

    print("\n=== Step 2: classify ===")
    backend = os.getenv("LLM_BACKEND", "ollama")
    llm = _build_llm(backend)
    classified_df = classify(clean_df, llm, CLASSIFY_CACHE)
    print(f"classified {len(classified_df)} rows (LLM_BACKEND={backend})")
    print(classified_df["sender_type"].value_counts().to_string())

    print("\n=== Step 3: load Knowledge Base ===")
    store = _build_kb_store()
    print(f"embedding model: {store.embedding_model_name} (requested: {store.embedding_model_name_requested})")
    entries = load_curated_kb(store, CURATED_KB_CSV)
    print(f"loaded {len(entries)} curated KB entries from {CURATED_KB_CSV}")
    topic_counts: dict[str, int] = {}
    for e in entries:
        topic_counts[e.topic] = topic_counts.get(e.topic, 0) + 1
    print(f"by topic: {topic_counts}")


if __name__ == "__main__":
    main()
