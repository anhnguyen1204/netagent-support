# CLAUDE.md — netAgent Support Intelligence Server

## What this project is

A self-contained, locally-runnable multi-agent server built **entirely from a historical
chat export** (`data/raw/output.csv` — columns: `create_at`, `userId`, `content`).
It does three things:

1. **Builds a Knowledge Base offline** from resolved problem→solution pairs found in the chat history.
2. **Serves live Q&A** — a user submits a question via REST API (later: chat platform), the
   system retrieves the closest verified solution and answers, or escalates if unsure.
3. **Monitors for spikes** — counts issue topics over time; if many users hit the same
   problem in a short window, fires an alert (email/webhook now; KTV chat DM later).

## Hard constraints (do not violate)

- **No access to company internal modules, APIs, or live server state.** There is no
  diagnostic agent that reads live workflow/credential/gateway status. Everything is
  built from the static CSV dataset and live user-submitted text only.
- **No fine-tuning.** Data is too small (~1,500 rows, mostly noise). Use LLM few-shot /
  embeddings, never train a classifier from scratch.
- **CPU only, possibly no LLM access at all.** The retrieval path (embeddings + vector
  search) must work with **zero LLM calls** as a fallback. LLM calls (if available) only
  improve phrasing/synthesis — they are never a hard dependency for the system to return
  an answer.
- **Read-only philosophy carries through even though there's no live server to read:**
  the system never claims to take an action. It answers and it alerts. Nothing else.

## Pluggable integration seams (stub now, swap later)

Three things are deliberately abstracted behind an interface so they can be swapped for
real company integrations later without touching the core logic:

| Seam | Interface | Stub today | Real version later |
|------|-----------|------------|---------------------|
| `Intake` | `intake.base.MessageSource` | REST endpoint / replay-from-CSV | netChat (Mattermost) webhook |
| `LLM` | `llm.base.LLMClient` | local Ollama model, or `NullLLM` (template-only) | netMind gateway |
| `Alerter` | `alerts.base.Alerter` | SMTP email / console log | netChat DM to KTV |

Every one of these is a small interface with one or two methods. Do not let any other
module import a concrete implementation directly — always go through the interface, so
swapping the stub for the real thing later is a one-file change.

## Repo structure (target)

```
netagent-support/
├── CLAUDE.md
├── docker-compose.yml
├── requirements.txt
├── data/
│   ├── raw/output.csv              # the dataset (user-provided)
│   ├── golden_set.csv              # hand-labeled eval set (~150-300 rows)
│   └── processed/                  # cleaned/intermediate parquet files
├── src/
│   ├── pipeline/                   # REGIME A — offline KB builder
│   │   ├── clean.py                # Bước 0: parse time, dedup, extract @mentions/entities
│   │   ├── classify.py             # Bước 1: sender/intent/topic labeling
│   │   └── synthesize_kb.py        # Bước 3: load hand-curated KB → vector DB
│   │                               # (thread.py / Bước 2 was built then removed — the KB
│   │                               #  is curated, not thread-extracted; see RESULTS.md)
│   ├── agents/                     # REGIME B — online agent graph (LangGraph)
│   │   ├── state.py                # shared graph state schema
│   │   ├── orchestrator.py         # B0: parse + route
│   │   ├── retrieval.py            # B1: KB search + CRAG-style relevance grading
│   │   ├── answerer.py             # B3: compose grounded answer
│   │   ├── critic.py               # B4: confidence scoring + gate
│   │   └── graph.py                # wires the above into a LangGraph graph
│   ├── monitor/
│   │   └── spike.py                # topic-frequency spike detector + alert trigger
│   ├── intake/
│   │   ├── base.py                 # MessageSource interface
│   │   ├── api_intake.py           # FastAPI endpoint implementation (stub)
│   │   └── replay_intake.py        # replays historical CSV through the system, time-ordered
│   ├── llm/
│   │   ├── base.py                 # LLMClient interface
│   │   ├── null_llm.py             # template-only fallback, zero LLM calls
│   │   ├── ollama_llm.py           # local model implementation
│   │   └── netmind_llm.py          # stub for company gateway (fill in if access granted)
│   ├── alerts/
│   │   ├── base.py                 # Alerter interface
│   │   ├── console_alerter.py      # prints to stdout/log
│   │   └── email_alerter.py        # SMTP implementation
│   ├── kb/
│   │   ├── store.py                # vector DB client wrapper (Qdrant)
│   │   └── schema.py                # KB entry schema (problem, solution, topic, source, confidence)
│   └── server.py                   # FastAPI app: wires intake → agent graph → response
├── eval/
│   ├── label_golden_set.py         # helper script to assist hand-labeling
│   └── run_eval.py                 # computes F1/precision/recall/retrieval metrics vs golden set
├── scripts/
│   ├── build_kb.py                 # CLI: run the full Regime A pipeline end-to-end
│   ├── run_server.py               # CLI: start the FastAPI server
│   └── replay_demo.py              # CLI: replay dataset through time, show spike alerts fire
└── tests/
    └── ...
```

## Conventions

- Python 3.11+, FastAPI, LangGraph, Qdrant (via `docker-compose`), `sentence-transformers`
  for embeddings (bge-m3), `pandas` for the offline pipeline.
- All Vietnamese text stays in UTF-8 throughout; no transliteration.
- Every agent/pipeline step takes typed input, returns typed output (pydantic models) —
  no passing raw dicts between stages.
- Every LLM call goes through `llm.base.LLMClient` — never call an API client directly
  from agent/pipeline code.
- Confidence scores are floats 0–1 on every classification/retrieval/answer step; the
  gate thresholds live in one config file, not scattered across code.
- Small, working increments. After each phase: run `scripts/build_kb.py` or
  `scripts/run_server.py` and confirm it still works before moving to the next phase.

## Testing/evaluation philosophy

Nothing is "done" without a number. Each phase has a definition-of-done tied to
`eval/run_eval.py` output against the golden set — see BUILD_PLAN.md.
