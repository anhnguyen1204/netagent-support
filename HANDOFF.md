# Handoff notes — swapping the stubs for real company integrations

Three integration points are deliberately behind interfaces. Each real version is a
**single new file implementing one interface**, plus wiring it in one place. Nothing
else in the codebase imports a concrete implementation, so none of the core logic
changes.

---

## 1. Intake — receive questions from netChat

- **Interface:** `src/intake/base.py` → `MessageSource` (one method: `listen()` yielding
  `IncomingMessage`).
- **Stub today:** the FastAPI `POST /ask` endpoint (`src/server.py`) and
  `src/intake/replay_intake.py` (replays the historical CSV).
- **Real version:** a netChat (Mattermost) webhook. Mattermost posts an outgoing-webhook
  payload when a user messages the support channel/bot; convert that payload into an
  `IncomingMessage(user_id, content, created_at)` and feed it to the same agent graph the
  `/ask` endpoint already uses (`app.state.graph` via `run_graph`).
- **What you need:**
  - A Mattermost **outgoing webhook** (or bot token + WebSocket) pointed at a new route,
    e.g. `POST /netchat/webhook`.
  - The bot's token to post replies back (Mattermost `/api/v4/posts`).
  - Decide reply policy per decision: `auto_reply` → post the answer; `suggest_to_staff`
    → post to a staff-only thread; `escalate` → DM the on-call KTV (reuses the Alerter).
- **Where to wire it:** add the route in `src/server.py`; no changes to `agents/` or
  `kb/`.

---

## 2. LLM — netMind gateway

- **Interface:** `src/llm/base.py` → `LLMClient` (one method: `complete(prompt) -> str`).
- **Stub today:** `NullLLM` (no LLM — template answers, rule-based classify) and
  `OllamaLLM` (local Ollama, implemented and working if a server is running).
- **Real version:** `src/llm/netmind_llm.py` already exists as a `NetMindLLM(api_url,
  api_key)` skeleton — fill in `complete()` to call the netMind gateway's completion
  endpoint.
- **What you need:**
  - netMind gateway **base URL** and an **API key** — set `NETMIND_API_URL` and
    `NETMIND_API_KEY` (already read by `src/config.py`).
  - The gateway's request/response shape (model name, prompt field, how the completion
    comes back) to implement `complete()`.
- **How to enable:** set `LLM_BACKEND=netmind`, then extend the `_build_llm()` helper in
  `src/server.py` (and `scripts/build_kb.py`) to return `NetMindLLM(...)` for that value
  — currently those helpers handle `ollama` and default to `NullLLM`.
- **Effect once enabled:** answers get LLM-polished, retrieval gains CRAG-style relevance
  grading, and classification uses few-shot instead of rules. None of it is required —
  everything already works on `NullLLM`.

---

## 3. Alerter — netChat DM to KTV

- **Interface:** `src/alerts/base.py` → `Alerter` (one method: `send(AlertRecord)`).
- **Stub today:** `ConsoleAlerter` (prints/logs) and `EmailAlerter` (SMTP, implemented).
- **Real version:** a `NetChatAlerter` that sends a Mattermost **direct message** to the
  on-call KTV when a spike is detected or a question escalates.
- **What you need:**
  - A bot token with permission to DM, and the target KTV user/channel id.
  - Map `AlertRecord` (topic, message, severity, triggered_at) to a Mattermost post.
- **Where to wire it:** implement the class in a new `src/alerts/netchat_alerter.py`, add
  an `ALERTER_BACKEND=netchat` branch to `_build_alerter()` in `src/server.py`. Both the
  spike monitor and the agent-graph escalation already call `Alerter.send()`, so both
  start routing to netChat automatically.

---

## Read-only guarantee

The system never takes an action on the platform — it only **answers** and **alerts**.
There is no live diagnostic path reading workflow/credential/gateway state; everything is
built from the static dataset + the curated KB + live user-submitted text. Keep this
property when adding the real integrations: the netChat intake reads messages and posts
answers/alerts; it must not mutate workflows, credentials, or system state.
