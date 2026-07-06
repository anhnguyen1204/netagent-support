"""FastAPI app: wires intake -> agent graph -> response.

At startup it builds the KB store (loading the curated KB, falling back to an in-memory
Qdrant if no server is reachable) and compiles the agent graph once, so each /ask request
just invokes the already-warm graph.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.agents.graph import build_graph, run_graph
from src.agents.memory import ConversationStore
from src.alerts.console_alerter import ConsoleAlerter
from src.alerts.email_alerter import EmailAlerter
from src.config import load_settings
from src.kb.store import KBStore
from src.llm.null_llm import NullLLM
from src.llm.ollama_llm import OllamaLLM
from src.monitor.spike import SpikeMonitor, run_spike_check
from src.pipeline.classify import _rule_based_classify
from src.pipeline.synthesize_kb import load_curated_kb

logger = logging.getLogger("netagent.server")

CURATED_KB_CSV = Path("data/curated_kb.csv")
STATIC_DIR = Path(__file__).resolve().parent / "intake" / "static"
SPIKE_CHECK_INTERVAL_SECONDS = 3600  # hourly tick for the live spike check


def _build_llm(settings):
    if settings.llm_backend == "ollama":
        return OllamaLLM(host=settings.ollama_host, model=settings.ollama_model)
    return NullLLM()


def _build_alerter(settings):
    if settings.alerter_backend == "email":
        import os
        return EmailAlerter(
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            to_address=os.getenv("ALERT_EMAIL_TO", ""),
        )
    return ConsoleAlerter()


def _build_store(settings) -> KBStore:
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
    except Exception:
        logger.warning(
            "Qdrant unreachable at %s:%s -- using in-memory store",
            settings.qdrant_host,
            settings.qdrant_port,
        )
        client = QdrantClient(":memory:")
    return KBStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
        embedding_model_fallback=settings.embedding_model_fallback,
        client=client,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    store = _build_store(settings)

    # If the collection is empty (e.g. the in-memory fallback, or a fresh server that
    # build_kb.py hasn't populated), load the curated KB so /ask has something to answer
    # from. On a real persistent Qdrant already populated by build_kb.py this is a no-op
    # re-upsert (idempotent by deterministic point id).
    load_curated_kb(store, CURATED_KB_CSV)

    llm = _build_llm(settings)
    alerter = _build_alerter(settings)
    app.state.graph = build_graph(store, llm, alerter, thresholds=settings.gate)
    app.state.llm_backend = settings.llm_backend

    # Spike monitor: each /ask question's topic is recorded live; an APScheduler job
    # periodically runs the spike check and fires the alerter on any detected spike.
    app.state.spike_monitor = SpikeMonitor()
    app.state.alerter = alerter
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_spike_check,
        "interval",
        seconds=SPIKE_CHECK_INTERVAL_SECONDS,
        args=[app.state.spike_monitor, alerter],
        id="spike_check",
    )
    scheduler.start()
    app.state.scheduler = scheduler
    app.state.conversations = ConversationStore()
    logger.info("agent graph + spike monitor ready (LLM_BACKEND=%s)", settings.llm_backend)

    yield

    scheduler.shutdown(wait=False)


app = FastAPI(title="netAgent Support Intelligence Server", lifespan=lifespan)


class HealthResponse(BaseModel):
    status: str


class AskRequest(BaseModel):
    question: str
    # optional: pass a stable id to make questions part of one multi-turn conversation.
    # omit it (or send null) for a stateless one-shot ask.
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    confidence: float
    decision: str
    source_thread_id: str | None = None
    session_id: str | None = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the minimal chat demo page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    # record the question's topic for live spike monitoring (rule-based, no LLM needed)
    _, _, topic, _ = _rule_based_classify(request.question)
    app.state.spike_monitor.record(topic, time.time() * 1000)

    # load prior turns for this session (empty if no session_id or first message)
    history = app.state.conversations.history(request.session_id)

    state = run_graph(app.state.graph, request.question, history=history)
    source = state.retrieved[0].entry.source_thread_id if state.retrieved else None

    # remember this turn so the next message in the session has context
    app.state.conversations.append(request.session_id, request.question, state.answer or "")

    return AskResponse(
        answer=state.answer or "",
        confidence=state.confidence if state.confidence is not None else 0.0,
        decision=state.decision or "escalate",
        source_thread_id=source,
        session_id=request.session_id,
    )
