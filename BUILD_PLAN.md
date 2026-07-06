# BUILD_PLAN.md — netAgent Support Intelligence Server

How to use this with Claude Code: work top to bottom, one phase per session (or per
sitting). Paste the "Session prompt" block for a phase directly into Claude Code. Each
phase ends with a **Definition of Done** — don't move to the next phase until it passes.
Read `CLAUDE.md` first (Claude Code will pick it up automatically if it's in the repo root).

---

## Phase 0 — Repo scaffold

**Goal:** empty-but-runnable skeleton, so every later phase has somewhere to land.

**Tasks**
- Create the repo structure exactly as listed in `CLAUDE.md`.
- `requirements.txt`: fastapi, uvicorn, langgraph, langchain-core, qdrant-client,
  sentence-transformers, pandas, pydantic, apscheduler, python-dotenv, pytest.
- `docker-compose.yml`: a single Qdrant service, persistent volume.
- Stub every file in `src/` with its class/function signatures and a `NotImplementedError`
  body — this gives Claude Code (and you) a map before any logic is written.
- `scripts/run_server.py` should start FastAPI and return a `{"status": "ok"}` health
  check at `/health`, even with everything else stubbed.

**Session prompt**
> Read CLAUDE.md. Scaffold the full repo structure described there. Every module should
> have correct imports, typed function/class signatures (use pydantic models where state
> is passed between components), and `raise NotImplementedError` bodies. Set up
> docker-compose.yml with a Qdrant service. Make `scripts/run_server.py` boot a FastAPI
> app with a working `/health` endpoint. Do not implement business logic yet.

**Definition of Done:** `docker compose up -d`, then `python scripts/run_server.py`,
then `curl localhost:8000/health` returns `{"status":"ok"}`.

---

## Phase 1 — Data cleaning + golden set (maps to Bước 0 + WP1/WP2 start)

**Goal:** clean dataframe + a hand-labelable golden set file.

**Tasks**
- `src/pipeline/clean.py`: load `data/raw/output.csv`, parse `create_at` (Unix epoch ms
  float → datetime), dedup selectively (drop exact `userId`+`content`+`create_at`
  duplicates only), extract `@mentions` via regex, extract entities (workflow URLs,
  IPs, node names) via regex, output a cleaned dataframe to
  `data/processed/clean_messages.parquet`.
- `scripts/build_kb.py` step 1: CLI command that runs `clean.py` and prints summary
  stats (row count before/after, # with mentions, # with entities).
- Sample ~200 cleaned rows (stratified — mix of short/long, with/without mentions) into
  `data/golden_set.csv` with empty columns: `sender_type`, `intent`, `topic`,
  `is_resolved` — ready for you to hand-label.

**Session prompt**
> Implement `src/pipeline/clean.py` per CLAUDE.md's Bước 0 spec: parse `create_at` as
> Unix epoch milliseconds, dedup only exact (userId, content, create_at) triples, extract
> `@mentions` and entities (workflow URLs matching netflow.viettel.vn/workflow/..., IPs,
> node-like tokens) into list columns. Write output to
> data/processed/clean_messages.parquet. Then write a sampling script that produces
> data/golden_set.csv: ~200 stratified rows with empty label columns (sender_type,
> intent, topic, is_resolved) for me to hand-label. Print before/after row counts and
> basic stats.

**Definition of Done:** `clean_messages.parquet` exists with sane row counts;
`golden_set.csv` exists with ~200 rows and empty label columns. **You then hand-label
this file before continuing** — this is the most important non-code task in the project.

---

## Phase 2 — Classification (Bước 1, WP2 baseline + WP3)

**Goal:** sender_type / intent / topic classifier, evaluated against your now-labeled
golden set.

**Tasks**
- `src/llm/base.py` (`LLMClient` interface: `complete(prompt: str) -> str`) +
  `src/llm/null_llm.py` (raises a clear "no LLM configured" with a keyword-rule
  fallback for sender_type only) + `src/llm/ollama_llm.py` (calls local Ollama if
  running) — wire via an env var `LLM_BACKEND=null|ollama|netmind`.
- `src/pipeline/classify.py`: regex pass for `system`/`noise` first (patterns from
  CLAUDE.md/data: "tham gia nhóm", "Rời nhóm", "đã ghim", 📢/🚨 announcements). Then,
  for the rest, one LLM few-shot prompt per message returning
  `{sender_type, intent, topics[], confidence}` as JSON. Cache LLM responses by message
  hash to avoid recomputation.
- `eval/run_eval.py`: load golden_set.csv (now labeled), run classify.py on those rows,
  compute accuracy/F1 per field, print a confusion matrix for sender_type.

**Session prompt**
> Implement the LLM interface and a NullLLM fallback (rule-based sender_type only, no
> intent/topic — return "unknown" with confidence 0). Implement OllamaLLM calling a
> local Ollama server. Implement src/pipeline/classify.py: regex-filter system/noise
> first (see CLAUDE.md data patterns), then for remaining rows call the configured
> LLMClient with a single few-shot prompt asking for sender_type, intent, topics[], and
> confidence as JSON for each message, with the intent and topic taxonomies listed in
> the per-step solution doc. Cache responses by content hash in
> data/processed/classify_cache.json. Then implement eval/run_eval.py to score
> classify.py's output against my hand-labeled data/golden_set.csv: print accuracy, F1,
> and a confusion matrix for sender_type.

**Definition of Done:** `python eval/run_eval.py` prints real numbers (not crashes) for
all three fields. Note the baseline numbers in a `RESULTS.md` — this is your WP2
deliverable.

---

## Phase 3 — Threading + resolution detection (Bước 2, WP4)

**Goal:** group messages into threads, detect resolved/unresolved status per thread.

**Tasks**
- `src/pipeline/thread.py`: build a graph where edges connect messages sharing an
  entity (workflow URL/IP/node), or an @mention within a time window, or
  close-in-time + high content similarity (use embeddings here — see Phase 4 for the
  embedding model choice, reuse it). Connected components = threads.
- Resolution detection: within each thread, scan for resolved/unresolved keyword
  signals (list in the per-step solution doc) in the customer's reply following a
  staff reply; handle the "📢 [DONE]" broadcast as a bulk-resolve signal correlated by
  time window.
- Extend `eval/run_eval.py` to score thread purity against a small hand-checked sample
  (~20 threads you manually verify) and resolution-detection accuracy.

**Session prompt**
> Implement src/pipeline/thread.py: build an undirected graph over cleaned messages
> with edges for (a) shared entity, (b) @mention reply within N minutes, (c) time
> proximity + content embedding similarity above a threshold. Connected components
> become thread_ids. Then implement resolution detection per thread: look at the first
> customer message after a staff message in the thread, check it against resolved/
> unresolved keyword lists (see Giai_phap_chi_tiet doc), and treat 📢 [DONE] broadcast
> messages as bulk-resolving all open threads referencing the same topic in the
> preceding N hours. Output thread_id, status, confidence, problem_summary (first
> customer message), solution (the resolving staff message) per thread.

**Definition of Done:** sample 20 threads, manually check them yourself, record rough
purity % in `RESULTS.md`. Doesn't need to be perfect — "good enough" per CLAUDE.md.

---

## Phase 4 — Knowledge Base (Bước 3, WP5 core)

**Goal:** resolved threads become a searchable, verified KB in Qdrant.

**Tasks**
- `src/kb/schema.py`: pydantic `KBEntry` (problem, solution, topic, source_thread_id,
  confidence, created_at).
- `src/kb/store.py`: thin Qdrant wrapper — `upsert(entries: list[KBEntry])`,
  `search(query: str, top_k: int) -> list[ScoredKBEntry]`. Embedding model:
  `sentence-transformers` with a multilingual model (bge-m3 if available, else
  `paraphrase-multilingual-mpnet-base-v2` as a lighter fallback that doesn't need a GPU).
- `src/pipeline/synthesize_kb.py`: for each resolved thread, optionally clean up the
  problem/solution text via the LLM (skip if NullLLM — just use the raw text), embed,
  upsert into Qdrant.
- `scripts/build_kb.py`: wire Phases 1–4 into one CLI command, end to end.

**Session prompt**
> Implement src/kb/schema.py (KBEntry pydantic model) and src/kb/store.py (Qdrant
> wrapper using sentence-transformers for embeddings — try to load bge-m3, fall back to
> paraphrase-multilingual-mpnet-base-v2 if unavailable). Implement
> src/pipeline/synthesize_kb.py: take resolved threads from Phase 3, optionally
> LLM-clean the problem/solution summaries, embed, and upsert into Qdrant via the
> store. Wire scripts/build_kb.py to run clean → classify → thread → synthesize_kb
> end-to-end with progress logging and a final summary (# KB entries, # by topic).

**Definition of Done:** `python scripts/build_kb.py` runs clean-to-KB with no LLM
required (NullLLM mode works), and `store.search("workflow mất publish", top_k=3)`
returns sane results.

---

## Phase 5 — Agent graph + API (Bước 4/6/7, WP5 Assist demo)

**Goal:** a FastAPI endpoint that takes a question, retrieves from KB, answers or
escalates — runnable with or without an LLM.

**Tasks**
- `src/agents/state.py`: pydantic `AgentState` (question, retrieved[], answer,
  confidence, decision).
- `src/agents/retrieval.py` (B1): call `kb.store.search`; if NullLLM, skip relevance
  grading and just threshold on embedding similarity score; if LLM available, grade
  relevance per result.
- `src/agents/answerer.py` (B3): if top result similarity/relevance is high, compose
  answer (template-based if NullLLM: "Vấn đề tương tự đã gặp: {problem} → Giải pháp:
  {solution} (nguồn: thread {id})"; LLM-polished if available).
- `src/agents/critic.py` (B4): confidence = top retrieval score (NullLLM) or
  LLM-graded (if available); apply gate thresholds from one config block.
- `src/agents/graph.py`: wire orchestrator → retrieval → answerer → critic as a
  LangGraph graph with conditional edges on the gate decision (auto-reply /
  suggest-to-staff / escalate).
- `src/intake/base.py` + `src/intake/api_intake.py`: `POST /ask {question: str}` →
  returns `{answer, confidence, decision, source_thread_id}`.
- `src/alerts/base.py` + `src/alerts/console_alerter.py`: called when decision is
  "escalate" — log/print a structured escalation record.

**Session prompt**
> Implement the LangGraph agent graph per CLAUDE.md Regime B (minus the diagnostic
> agent — it's removed, no live server access). Build: AgentState schema, retrieval
> node (calls kb.store.search, grades relevance via LLM if available else thresholds on
> raw similarity), answerer node (composes a grounded answer citing the source thread —
> template-based fallback if no LLM), critic node (confidence score + gate: high =
> auto-reply, medium = suggest-to-staff with a flag, low = escalate). Wire as a
> LangGraph StateGraph with conditional routing on the gate decision. Expose via FastAPI
> POST /ask. On "escalate" decisions, call the Alerter interface (console implementation
> for now).

**Definition of Done:** `python scripts/run_server.py`, then `curl -X POST
localhost:8000/ask -d '{"question":"workflow của em bị mất publish"}'` returns a
grounded answer citing a real source thread from your data, even with `LLM_BACKEND=null`.

---

## Phase 6 — Spike monitor + alerting (WP5 extension)

**Goal:** detect when many users hit the same topic in a short window; fire an alert.

**Tasks**
- `src/monitor/spike.py`: maintain a rolling count of topic occurrences per time
  bucket (e.g. 1-hour buckets); flag a spike when count exceeds
  `mean + k * stddev` of recent history, or a simple absolute jump threshold for the
  cold-start period where there's no history yet.
- Wire as an APScheduler background job inside `server.py` for live mode, OR as a
  step inside `scripts/replay_demo.py` for the historical replay.
- `scripts/replay_demo.py`: replay `clean_messages.parquet` in timestamp order through
  the classify + spike monitor pipeline, printing each alert as it would have fired
  historically — this should visibly catch the real incident waves already in your
  dataset (vmail blocking, mass "mất publish" reports).

**Session prompt**
> Implement src/monitor/spike.py: a rolling topic-frequency counter over configurable
> time buckets, flagging a spike via mean+k*stddev (with a simple absolute-count
> fallback during cold start). Wire it as an APScheduler job in server.py for live mode.
> Then implement scripts/replay_demo.py: replay data/processed/clean_messages.parquet
> in timestamp order through classify + spike monitor, calling the Alerter on each
> detected spike, and print a timeline of alerts. This is the main demo script — it
> should visibly catch the known incident clusters in the dataset (vmail trigger
> blocking, mass publish-loss reports).

**Definition of Done:** `python scripts/replay_demo.py` prints at least one spike alert
that corresponds to a real incident broadcast (📢) already present in the data — that's
your validation that the detector works, without needing ground truth labels.

---

## Phase 7 — Polish for demo / handoff

**Tasks**
- Minimal web page (`src/intake/static/index.html`) — a chat box hitting `POST /ask`,
  for a live demo instead of curl.
- `RESULTS.md` — consolidate all eval numbers from every phase into one summary table.
- `README.md` — how to run: `docker compose up -d`, `pip install -r requirements.txt`,
  `python scripts/build_kb.py`, `python scripts/run_server.py`, `python
  scripts/replay_demo.py`.
- Confirm every integration seam (`Intake`, `LLMClient`, `Alerter`) has a clearly
  marked stub and a one-paragraph note on what the real version would need (API
  credentials, endpoint URL) — this is what you hand the platform team when/if access
  is granted.

**Definition of Done:** a fresh clone + the four commands above gets a stranger to a
working demo with zero company access required.

---

## Notes for working with Claude Code across sessions

- Keep CLAUDE.md updated as you go — if you change a schema or a convention, update it
  immediately so later sessions stay consistent.
- After each phase, commit. Small commits make it easy to roll back a session that goes
  sideways.
- If a session stalls or produces something you don't like, it's cheaper to revert and
  re-prompt more precisely than to argue with the diff.
- The golden set (Phase 1) is the one task that's on you, not Claude Code — budget real
  time for it, the whole project's credibility rests on it.
