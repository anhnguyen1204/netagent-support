# RESULTS.md — netAgent Support Intelligence Server

Consolidated eval numbers from every phase. Updated as each phase completes.

## Phase 1 — Data cleaning + golden set

- Raw rows: 1497 (52 rows had null `content`, dropped)
- Rows after dropna + exact-duplicate dedup (`userId`+`content`+`create_at`): 1325
- Rows with `@mentions` extracted: 266
- Rows with entities (URLs/IPs/node names) extracted: 56
- `data/golden_set.csv`: 200 rows, rebuilt (v2) with content-aware sampling instead of
  pure length/mention stratification — targets realistic composition rather than
  noise's natural ~35% population share. `sender_type`: 10 system, 95 customer,
  65 staff, 30 unknown. `topic`: node_feature (44), connection_access (23),
  workflow_run/workflow_publish (~11-12 each), llm_model/datatable/email/credential/
  infra_incident all represented; `other`+`none` down to ~36% combined (was ~65% in v1).
  Labels applied via direct reading/judgment against `data/TAXONOMY.md`, cross-checked
  against raw context — **draft, pending your review**, per project convention that
  golden-set ground truth needs independent verification before Phase 2 scores against it.
- `is_resolved` stays mostly `unknown` by design: tracing resolution from raw chat
  context proved unreliable during v1 review (messages are heavily timestamp-bucketed
  and interleaved across simultaneous conversations) — real resolution detection is
  deferred to Phase 3's thread-graph output, not single-message inspection.

Note: 6 rows in the raw CSV have a message truncated mid-UTF8-character during
export; `clean.py` decodes with `errors="replace"` rather than failing the whole
pipeline over a handful of corrupted trailing characters.

## Phase 2 — Classification (baseline, `LLM_BACKEND=null`)

`python eval/run_eval.py` against the 200-row golden set, rule-based `NullLLM` classifier
(no LLM calls — regex/keyword signals only):

| field | accuracy | macro F1 |
|---|---|---|
| sender_type | 0.485 | 0.588 |
| intent | 0.410 | 0.344 |
| topic | 0.450 | 0.442 |

`sender_type` confusion matrix shows the main failure mode: the rule-based classifier
over-predicts `unknown` (recall 0.967, precision 0.246) — 53/95 real `customer` rows and
36/65 real `staff` rows got no matching regex signal and fell through to `unknown`.
`system` detection is perfect (1.00/1.00), since the join/leave/pin patterns are exact
string matches. This is the expected shape of a zero-LLM baseline per CLAUDE.md's hard
constraint (retrieval must work with zero LLM calls) — it is a floor, not a target;
`LLM_BACKEND=ollama` (few-shot classification) should score meaningfully higher and is
implemented in `src/llm/ollama_llm.py`, untested here since Ollama is not installed in
this environment.

## Phase 3 — Threading + resolution detection

`src/pipeline/thread.py`: graph over non-`system` messages (system events explicitly
excluded — batches of near-identical join/leave messages within the same minute were
spuriously linked by embedding similarity into fake giant "threads" before this
exclusion was added) with edges for shared entity (URL/IP/node token), @mention-reply
within 30 min, and time-proximity + embedding similarity (`paraphrase-multilingual-
mpnet-base-v2`, threshold 0.75, 60 min window; messages under 8 chars excluded from
similarity matching — near-empty content produces near-random embeddings).

**Bug found and fixed: @mention edges were silently contributing zero signal.**
`@mentions` in message content are human-readable handles (e.g. `"@hoaint44"`), but
`user_id` is an opaque hash (e.g. `"xryqg8ydrfgeumkks1ptgxtway"`) — the two are never
equal, so the original `_mention_edges` (matching `mention.lstrip("@")` directly
against `user_id`) matched nothing, ever, across the whole dataset (verified: 0 staff
mention-replies found before the fix). Fix: the "X tham gia nhóm." / "X được thêm vào
nhóm bởi Y." system messages incidentally record each user's handle next to their real
`user_id` in the same row — mined that into a handle→user_id lookup
(`_build_handle_to_user_id`, 372 handles recovered, 8/8 real staff-mention targets
resolved correctly) and used it to translate mentions before matching.

Effect: multi-message threads went from 31 → 67 (20→41 pairs, 11→26 with 3+ messages),
and threads containing BOTH a staff and a customer message (the ones that can actually
become KB entries) went from **2 → 9**.

**Hand-checked all 26 threads with 3+ messages** (BUILD_PLAN asks for ~20 — the full
2-message set (41) was not exhaustively re-reviewed after the mention-edge fix, but
spot-checks did not surface new failure modes beyond what's listed below):
- ~15-17/26 topically coherent (~60-65% purity) — consistent with the pre-fix rate,
  confirming the fix added real signal without degrading overall thread quality
- Genuinely strong examples: a "netflow bị 502" report correctly linked to staff's
  "load lại giúp, do tràn memory" reply; a "VMail access mở lại chưa?" question linked
  to staff's status update; an LLM-timeout report linked to staff's infra-overload
  explanation
- Remaining false-link causes unchanged from before the fix: generic word overlap,
  coincidental short acknowledgments to different people, and broadcasts
  (`@all`-addressed announcements) occasionally still linked via similarity edges to
  unrelated nearby messages — worth tightening further in a future pass, but "good
  enough" per CLAUDE.md's explicit bar for this phase.

**Bulk-resolve requires topic match, not just time proximity** (first version matched
any thread within 6h of a `[DONE]` broadcast regardless of subject, sweeping in ~37
unrelated threads on manual review — e.g. bare connection requests near an unrelated
incident). After requiring the thread's classified `topic` to match the broadcast's
topic: only 2 real `[DONE]` broadcasts exist in the dataset (both correctly classified
`infra_incident`), and only 1 preceding thread in their 6h windows shares that topic —
so exactly 1 thread gets bulk-resolved. This is a small, honest number rather than a
bug: it's bounded by (a) how rarely `[DONE]` broadcasts actually appear in this dataset
and (b) Phase 2's classifier recall on `topic` (per its eval numbers, real incident
reports likely get misclassified as `other` rather than `infra_incident` at a
meaningful rate) — a better Phase 2 classifier would likely surface more matches here
without any change to the threading logic itself.

## Phase 4 — Knowledge Base

`src/kb/schema.py` (`KBEntry`/`ScoredKBEntry`) + `src/kb/store.py` (Qdrant wrapper,
tries `BAAI/bge-m3` first — succeeded in this environment, 1024-dim — falls back to
`paraphrase-multilingual-mpnet-base-v2` (768-dim) only if bge-m3 fails to load).

**The KB is built from a hand-curated set, not auto-extracted from conversation
threads.** Auto-extraction was fully built and evaluated first (thread grouping →
resolution detection → problem/solution pairing), and along the way surfaced and fixed
several real bugs — but the honest conclusion was that this dataset does not contain
enough clean, resolved, single-topic exchanges for auto-extraction to be worth keeping.
The auto-extraction path was therefore **removed** from `synthesize_kb.py` and
`build_kb.py` (per user decision); `build_kb.py` is now clean → classify → load curated
KB. (Conversation threading itself remains in `src/pipeline/thread.py` for the Phase 6
spike monitor — only the thread→KB extraction was dropped.)

### What auto-extraction taught us before removal (kept for the record)

- **`store.search()` machinery verified sound**: manually-seeded entries ranked
  correctly (`store.search("workflow mất publish")` → matching entry at 0.78, clear
  separation from unrelated topics).
- Two upstream bugs found and fixed in Phase 3 while chasing KB yield (documented in
  Phase 3 above): the silent @mention-edge namespace mismatch (2 → 9 staff+customer
  threads), and `detect_resolution` requiring explicit customer confirmation.
- Four KB-quality passes on the extraction (exclude broadcasts from edges; pair by
  embedding similarity not "first staff message"; confidence tiers + a 0.5 storage bar;
  trim UTF-8 truncation) improved precision but drove yield down to **1 trustworthy
  auto-extracted entry** — because the deeper limit is Phase 3 over-merging threads at
  the graph level, which no pairing heuristic downstream can undo. Measured directly:
  a real LLM-timeout problem→solution pair scores 0.63 cosine, but a real datalake pair
  only 0.37 (below a mispaired 0.44), so embedding similarity alone cannot cleanly
  separate good pairings from bad on this data. That 1-entry result is what motivated
  going curated-only.

### Curated KB (`data/curated_kb.csv`, 39 entries)

39 hand-curated problem→solution pairs, each read directly out of the raw chat and
grounded in a real staff answer (not invented). Marked `source_thread_id="curated"`,
confidence 0.95. Topic coverage: node_feature ×12, llm_model ×6, connection_access ×5,
datatable ×4, credential ×3, workflow_run ×3, workflow_publish ×2, email ×2,
infra_incident ×2 — all 9 non-trivial topics represented.

**`store.search()` retrieval verified across 12 diverse real queries — every one
returns the correct topic + a relevant solution as the top hit**, scores 0.61–0.88:
- "workflow của em bị mất publish" → 0.71 (workflow_publish, vmail/unpublish)
- "vmail trigger không chạy" → 0.82 (email)
- "credential token hết hạn" → 0.61 (credential, netMind token-refresh)
- "LLM báo lỗi timeout 504" → 0.76 (llm_model)
- "import csv vào datatable bị lỗi header" → 0.88 (datatable, "change `;` to `,`")
- "không vào được netflow bị 502" → 0.79 (connection_access, "reload, memory overflow")
- "muốn vẽ biểu đồ chart trong netflow" → 0.61 (node_feature, Report Engine)
- "gửi nhiều ảnh trong 1 tin nhắn netchat" → 0.78 (node_feature)
- "model Qwen không xử lý ảnh" → 0.77 (llm_model)
- "xin mở kết nối tới datalake" → 0.72 (connection_access)
- "hệ thống đang bảo trì" → 0.65 (infra_incident)
- "bot chỉ trả lời mình không trả lời người khác" → 0.66 (node_feature)

`docker compose up -d` was not available to test against (Docker not installed in this
environment); `KBStore` falls back to Qdrant's `:memory:` client automatically when the
configured host:port is unreachable, verified working in `build_kb.py`.

## Phase 5 — Agent graph + API

LangGraph `StateGraph`: orchestrate → retrieve → answer → critic, with a conditional
edge on the gate decision (`src/agents/*`, wired in `graph.py`). Runs with
`LLM_BACKEND=null` (no LLM required):
- **retrieval** (`retrieval.py`) calls `store.search`, drops results below raw cosine
  0.35; with a real LLM it additionally does CRAG-style yes/no relevance grading per
  result (skipped on NullLLM).
- **answerer** (`answerer.py`) composes a grounded template answer citing the source
  ("Vấn đề tương tự đã gặp: … → Giải pháp: … (nguồn: …)"); LLM-polishes if available.
- **critic** (`critic.py`) confidence = retrieval score × the matched entry's stored
  confidence (a strong match to a weak entry shouldn't auto-reply), then gates via
  `GateThresholds` (`src/config.py`, single config block): ≥0.75 auto_reply, ≥0.45
  suggest_to_staff, else escalate.
- **escalate** fires the `Alerter` (ConsoleAlerter by default; EmailAlerter implemented
  for `ALERTER_BACKEND=email`) with a structured record.

`src/server.py` builds the store + curated KB + compiled graph once at startup
(FastAPI lifespan) and serves `POST /ask {question}` → `{answer, confidence, decision,
source_thread_id}`.

**Definition of Done met — live `curl` tests (`LLM_BACKEND=null`):**
- `POST /ask {"question":"workflow của em bị mất publish"}` → grounded answer (vmail/
  unpublish solution) citing source, decision `suggest_to_staff` (conf 0.69).
- Near-exact datatable query → `auto_reply` (conf 0.77).
- "how do I cook beef pho" (irrelevant) → no KB match, decision `escalate`, ConsoleAlerter
  fired.

Note on thresholds vs embedding model: the server defaults to `bge-m3`, whose cosine
scores run more conservative than `mpnet` (a datatable paraphrase scores 0.76 on bge-m3
vs 0.88 on mpnet). Net effect: casual paraphrases land in `suggest_to_staff`, only close
matches reach `auto_reply` — a deliberately safe default for a support bot (human reviews
before auto-send). Thresholds live in one config block and are easy to retune per model.

## Phase 6 — Spike monitor + alerting

`src/monitor/spike.py`: `SpikeMonitor` keeps rolling per-topic counts in time buckets
(default 1 day — this dataset averages ~7 msgs/hour but is bursty, and per-topic hourly
counts max out at ~3, so daily is where real incident waves show up). A bucket flags a
spike when its count exceeds `mean + k·stddev` (k=2) of that topic's prior buckets
(zero-count days included in the baseline), or an absolute threshold (≥4) during cold
start before enough history exists.

**Bug found and fixed during the replay run**: the first version fired **43 alerts**,
most of them "spike of 1 report" — because a topic whose normal daily volume is ~0 has
`mean 0.1 + 2·0.4 stddev < 1`, so even a single report cleared the statistical bar.
Added a hard `min_spike_count = 3` floor (a bucket with <3 reports is never a spike,
whatever the baseline). This dropped it to **9 clean, meaningful alerts**.

`src/intake/replay_intake.py` (`ReplayIntake`, time-ordered replay) + `scripts/replay_demo.py`
replay all 1325 messages in timestamp order through classify + the spike monitor,
running a check each time the day-bucket advances (mirroring the live scheduler tick).

**Definition of Done met.** The 9 alerts are the real incident waves in the data — the
"mất publish" cluster (`workflow_publish` spikes of 5/3/3 on 2026-04-10/13/14 UTC
buckets), plus `node_feature` and `connection_access` waves. Crucially, the
**`node_feature` spike on 2026-04-09 (5 reports) coincides with a real 📢/🚨 incident
broadcast on the same day** — i.e. the detector fires on an incident day that also
produced a broadcast, which is exactly the historical validation asked for. (Note:
buckets are UTC-epoch floored; comparing spike days and broadcast days both in UTC is
the correct apples-to-apples check — a naïve local-time display can look a day off.)

**Live mode wired into `server.py`**: an APScheduler `BackgroundScheduler` runs
`run_spike_check` hourly; each `/ask` question's topic (rule-based classification, no LLM
needed) is recorded into the monitor. Verified via `TestClient`: scheduler starts, a
posted question records its topic (`workflow_publish: 1`), scheduler shuts down cleanly
on app shutdown.

## Multi-turn conversation (`/ask` with `session_id`)

Added `ConversationStore` (`src/agents/memory.py`) — per-session turn history — and made
`retrieve`/`answer` history-aware so follow-ups work ("nguyên nhân của nó là gì?" resolves
against the prior turn). Requires an LLM backend (history-aware query rewriting needs it);
stateless without one.

**Iteration to make it deterministic, not flaky**, tracked through real user testing:

1. **First bug**: `_mention_edges`-style naive anchoring (folding raw history text into
   the search query) inflated scores uniformly, dragging off-topic follow-ups
   ("thời tiết hôm nay") above the LLM-trust threshold — the system answered questions it
   should have escalated.
2. **Fix**: separated RECALL (anchored search finds the right candidate) from CONFIDENCE
   (candidates re-scored against the bare query, max of the LLM rewrite and the raw
   question, so an off-topic follow-up's confidence reflects the bare match, not the
   inflated anchored one). Off-topic follow-ups now reliably escalate.
3. **Second bug** (found via live user testing, not initial test cases): a vague
   follow-up late in a conversation ("cách xử lý lỗi này") picked the wrong
   *similar*-but-different KB entry (`workflow_run` instead of `workflow_publish`)
   ~25% of the time (2/8 in testing). Root cause: the anchor used only the
   `ANCHOR_RECENT_TURNS` (2) most recent questions — in a 3+-turn conversation this
   drops the *first* turn, which states the original problem and carries the strongest
   topic signal; later turns ("nguyên nhân là gì") are themselves generic and anchor
   weakly.
4. **Fix**: `_anchor_text` always includes the first turn regardless of conversation
   length, in addition to the most recent turns. Measured improvement: 6/8 → 8/10
   correct on the repro case (75% → 80%). Wrong picks now correlate with a visibly
   lower confidence score (~0.48) vs correct ones (~0.62-0.63) — a real, if imperfect,
   confidence signal that a future pass could use to ask for clarification instead of
   guessing (not implemented; diminishing returns on further tuning a 7B model's
   inherent rewrite noise, documented here rather than chased further).

**Default LLM model raised 3B → 7B** (`qwen2.5:7b`) partway through this work — its
rewrites/grading are markedly more consistent, at the cost of ~2-4s/answer instead of
~1-2s on CPU.

**Known residual limitation**: vague, topic-ambiguous follow-ups late in a long
conversation have a real (~20%) chance of retrieving a similar-but-wrong KB entry. Not
observed for follow-ups that retain any specific wording from the original problem.

## Answer style (LLM boilerplate)

User-reported: every LLM-synthesized answer opened with "Chào bạn," and closed with
generic Vietnamese customer-service filler ("Chúc bạn một ngày tốt lành!", "Nếu cần hỗ
trợ thêm...") regardless of the question. `POLISH_PROMPT` had asked for "concise,
friendly" but never forbade greetings/sign-offs, so the 7B model defaulted to
boilerplate it was evidently trained on. Fixed by making the prompt explicitly forbid
opening with a greeting or closing with a sign-off, and capping length to 1-4 sentences
unless the solution has multiple steps. Verified live against Ollama across several
questions — direct content only, no wrapper text.

While testing this, observed one instance (1/19 sampled calls) of Chinese-character
leakage mid-answer from `qwen2.5:7b` — not reproduced in a larger follow-up sample
(0/12), likely a rare sampling artifact rather than something the prompt reliably
triggers. Not defended against in code; noted here as a residual risk to watch for if
it recurs at a higher rate.