# data/raw/

Place the chat export here as `output.csv` (columns: `create_at`, `userId`, `content`).

The real dataset is **not committed** to this repo — it contains real employee handles,
IPs, and internal URLs. Provide your own `output.csv` to run the offline pipeline
(`scripts/build_kb.py`). The live Q&A server does **not** need it — the Knowledge Base is
built from `data/curated_kb.csv`, which is committed.
