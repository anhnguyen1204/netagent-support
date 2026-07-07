# netAgent Support Intelligence Server

A self-contained, locally-runnable support-automation system built entirely from a
historical chat export (`data/raw/output.csv` — ~1,500 Vietnamese support messages for
the netAgent/netFlow platform). It does three things:

1. **Serves live Q&A** — submit a question via REST; the system retrieves the closest
   verified solution from a Knowledge Base and answers, or escalates if unsure.
2. **Builds the Knowledge Base offline** from a curated set of real problem→solution
   pairs (see [Knowledge Base](#knowledge-base) for why curated rather than auto-mined).
3. **Monitors for spikes** — counts issue topics over time; if many users hit the same
   problem in a short window, it fires an alert.

Everything runs **CPU-only** and **without an LLM** — the retrieval path uses embeddings
+ vector search with zero LLM calls. An LLM (Ollama) only improves answer phrasing if
available; it is never required.

---

## Quick start

```bash
# 1. install dependencies (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (optional) start Qdrant for a persistent vector DB.
#    If skipped, the app falls back to an in-memory Qdrant automatically.
docker compose up -d

# 3. build the Knowledge Base (clean → classify → load curated KB)
PYTHONPATH=. python scripts/build_kb.py

# 4. run the server, then open http://localhost:8000 in a browser
PYTHONPATH=. python scripts/run_server.py

# 5. (demo) replay the historical data through the spike monitor
PYTHONPATH=. python scripts/replay_demo.py
```

> **Note:** commands are run from the repo root with `PYTHONPATH=.` so the `src`
> package resolves. `docker compose up -d` is optional — without it, the KB lives in an
> in-memory Qdrant that is rebuilt on each server start (fine for a demo).

### Enabling LLM reasoning (optional)

By default (`LLM_BACKEND=null`) answers are the KB solution shown via a template, and
the escalate/answer decision is purely the embedding similarity score. Enabling a local
LLM adds two things: **relevance grading** (the LLM rejects off-topic KB hits, so a
question like "explain netAgent" correctly escalates instead of returning an unrelated
entry) and **answer synthesis** (the solution is rephrased conversationally).

```bash
# 1. install Ollama (https://ollama.com) and pull a small multilingual model
ollama serve &                 # start the Ollama server (keep it running)
ollama pull qwen2.5:7b

# 2. run the app pointed at Ollama
LLM_BACKEND=ollama OLLAMA_MODEL=qwen2.5:7b PYTHONPATH=. python scripts/run_server.py
```

On CPU each `/ask` takes ~2–4s with the default `qwen2.5:7b` (a smaller 3B model is
faster, ~1–2s, but its query rewrites and relevance judgments are noticeably noisier for
multi-turn). Strong-similarity KB matches are trusted on the embedding score alone (not
vetoed by the LLM), so a small model's occasional misjudgment can't discard a clearly-
relevant answer — see `src/agents/retrieval.py` (`LLM_GRADE_TRUST_SCORE`).

### Ask a question

Open **http://localhost:8000** for the chat UI, or use curl:

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"workflow của em bị mất publish"}'
```

**Multi-turn conversation:** pass a stable `session_id` to make questions part of one
conversation — the system carries context so follow-ups work (a follow-up like "nguyên
nhân của nó là gì?" is resolved against the earlier turn). Requires an LLM backend for
the history-aware query rewriting; omit `session_id` for stateless one-shot asks. The web
UI sets a session automatically per page load.

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"workflow của em bị mất publish","session_id":"abc"}'
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"nguyên nhân của nó là gì?","session_id":"abc"}'   # understands "nó"
```

Response:

```json
{
  "answer": "Vấn đề tương tự đã gặp: … → Giải pháp: … (nguồn: curated)",
  "confidence": 0.69,
  "decision": "suggest_to_staff",
  "source_thread_id": "curated"
}
```

`decision` is one of `auto_reply` (high confidence), `suggest_to_staff` (medium — a human
should confirm), or `escalate` (no confident match — an alert is fired).

---

## Running the tests

```bash
pytest                  # all 59 tests (~12s; the slow ones load the embedding model)
pytest -m "not slow"    # fast tier only (~2.5s, no embeddings/network)
```

---

## Architecture

Two regimes:

- **Regime A (offline, `scripts/build_kb.py`)**: `clean` → `classify` → load the curated
  Knowledge Base into Qdrant. Run once (or when the KB changes).
- **Regime B (online, `scripts/run_server.py`)**: a FastAPI `/ask` endpoint driving a
  LangGraph agent: `orchestrate → retrieve → answer → critic`, which gates the answer
  into auto-reply / suggest-to-staff / escalate. A background spike monitor
  (APScheduler) watches question topics live.

```
src/
├── pipeline/     clean.py, classify.py, synthesize_kb.py   (Regime A)
├── agents/       orchestrator, retrieval, answerer, critic, graph, state   (Regime B)
├── kb/           schema.py (KBEntry), store.py (Qdrant + embeddings)
├── monitor/      spike.py (topic-frequency spike detector)
├── intake/       base.py + api_intake / replay_intake + static/index.html
├── llm/          base.py + null_llm / ollama_llm / netmind_llm
├── alerts/       base.py + console_alerter / email_alerter
├── config.py     gate thresholds + settings (one place)
└── server.py     FastAPI app
```

### Knowledge Base

The KB is built from **`data/curated_kb.csv`** — 39 hand-curated problem→solution pairs,
each read directly from the raw chat and grounded in a real staff answer.

Automated extraction (grouping messages into conversation threads, detecting resolution,
pairing problem↔solution) was fully built and evaluated first, but this dataset contains
almost no clean, resolved, single-topic exchanges — it is mostly one-off messages and
questions with no traceable reply. Auto-extraction topped out at 1 trustworthy entry, so
the thread-extraction path was removed and the KB is curated. The full analysis is in
[`RESULTS.md`](RESULTS.md).

---

## Configuration

Copy `.env.example` and adjust as needed. Key variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_BACKEND` | `null` | `null` (no LLM, template answers) / `ollama` / `netmind` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | embedding model; falls back to `paraphrase-multilingual-mpnet-base-v2` if it can't load |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | vector DB (in-memory fallback if unreachable) |
| `ALERTER_BACKEND` | `console` | `console` / `email` |

Gate thresholds (auto-reply / suggest / escalate cutoffs) live in one place:
`src/config.py` → `GateThresholds`.

---

## Integration seams (stubbed now, swap later)

Three seams are behind interfaces so real company integrations drop in as a one-file
change — nothing else imports a concrete implementation directly. See
[`HANDOFF.md`](HANDOFF.md) for what each real version needs.

| Seam | Interface | Stub today | Real version |
|------|-----------|------------|--------------|
| Intake | `intake.base.MessageSource` | REST `/ask` + replay-from-CSV | netChat (Mattermost) webhook |
| LLM | `llm.base.LLMClient` | `NullLLM` / local Ollama | netMind gateway |
| Alerter | `alerts.base.Alerter` | console log / SMTP email | netChat DM to KTV |
