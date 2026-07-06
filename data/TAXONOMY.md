# Golden Set Labeling Taxonomy

Fixed value menus for hand-labeling `data/golden_set.csv`. Derived from patterns in the
actual data. Use these exact string values so `classify.py` and `eval/run_eval.py` agree.

## `sender_type`
Who/what produced the message.

| value | meaning |
|-------|---------|
| `system` | Auto-generated group events: "tham gia nhóm", "Rời nhóm", "được thêm vào nhóm bởi…", "đã ghim/bỏ ghim một tin nhắn", "pinned/unpinned a message", bare join links. |
| `customer` | A user reporting a problem, asking a question, or replying as the person being helped. |
| `staff` | netAgent/support team member answering, troubleshooting, or broadcasting status. |
| `unknown` | Genuinely ambiguous — can't tell customer vs staff from text alone (e.g. a bare "@mention" with no content, or "ôi sợ quá"). |

## `intent`
What the message is trying to do. (Only meaningful for `customer`/`staff`; for `system` use `none`.)

| value | meaning |
|-------|---------|
| `report_problem` | Reporting something broken: "bị lỗi", "không chạy", "mất publish". |
| `ask_question` | Asking how to do something / whether something is possible: "có thể … không?", "làm sao". |
| `request_access` | Asking to be connected/added/granted: "xin kết nối", "add giúp", "xin API key". |
| `provide_solution` | Staff (or peer) giving a fix, workaround, or explanation. |
| `acknowledge` | Thanks / confirmation / social closure: "em cảm ơn", "Dạ vâng", "ok ạ". |
| `status_update` | Broadcast or progress note: "đang kiểm tra", "📢 [DONE]", "sẽ fix và thông báo". |
| `none` | System messages / pure noise with no support intent. |

## `topic`
Primary subject. Pick the single best fit. (For `system`/noise use `none`.)

| value | meaning |
|-------|---------|
| `workflow_publish` | Publish/unpublish state, flow turning off by itself, "mất publish". |
| `workflow_run` | Flow execution: didn't run, ran twice, manual run, scheduling. |
| `credential` | Credential create/save errors, auth/token issues, "Authen sai". |
| `node_feature` | Questions/issues about a specific node (AI Agent, OCR, Code, NetChat, Tableau, email/IMAP). |
| `email` | Email get/send specifically (IMAP, filters, password) — use when email is the core subject. |
| `datatable` | Datatable storage limits, import csv/xlsx errors. |
| `llm_model` | Model selection/limits, AI gateway overload, LLM infra errors. |
| `connection_access` | Connecting netAgent to a DB/service, N8N access, account connection. |
| `infra_incident` | Platform-wide outages: K8s, hạ tầng, gateway down (usually staff broadcasts). |
| `other` | Real support content that fits none of the above. |
| `none` | System messages / pure noise. |

## `is_resolved`
Per-message signal of whether the *issue under discussion* is resolved. Most rows are not
individually resolvable — use `unknown` liberally; reserve `true`/`false` for clear cases.

| value | meaning |
|-------|---------|
| `true` | Message indicates the problem is fixed: "📢 [DONE]", "sáng public lại rồi", customer "đã được rồi cảm ơn". |
| `false` | Message indicates still-broken / unresolved: "vẫn bị lỗi", "vẫn chưa vào được". |
| `unknown` | No resolution signal (questions, social chatter, system messages, mid-conversation). |