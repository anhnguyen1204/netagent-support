"""CLI: replay dataset through time, show spike alerts fire.

Replays data/processed/clean_messages.parquet in timestamp order through classify +
spike monitor, calling the Alerter on each detected spike, and prints a timeline of
alerts. This is the historical validation of the monitor: it should surface the real
incident-wave days already present in the data (e.g. the workflow "mất publish" cluster,
the vmail/infra incident days that coincide with 📢 broadcasts).

Classification now requires a live LLM backend (Ollama) -- the first run over all
~1300 messages makes one LLM call per uncached message; subsequent runs hit
data/processed/classify_cache.json (keyed by content hash) and are fast.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.alerts.console_alerter import ConsoleAlerter
from src.llm.ollama_llm import OllamaLLM
from src.monitor.spike import SpikeMonitor, run_spike_check
from src.pipeline.classify import classify

CLEAN_PARQUET = Path("data/processed/clean_messages.parquet")
CLASSIFY_CACHE = Path("data/processed/classify_cache.json")


def _fmt(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d")


def main() -> None:
    print("=== Replay demo: historical spike detection ===\n")

    llm = OllamaLLM(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    )
    df = pd.read_parquet(CLEAN_PARQUET)
    df = classify(df, llm, CLASSIFY_CACHE).sort_values("created_at")
    print(f"replaying {len(df)} messages in timestamp order\n")

    monitor = SpikeMonitor(bucket_minutes=24 * 60, k=2.0, cold_start_abs_threshold=4)
    alerter = ConsoleAlerter()

    # Feed messages in time order; run a spike check whenever the day-bucket advances so
    # a spike is reported as soon as its bucket is complete (mirrors how the live
    # APScheduler job in server.py would tick).
    current_bucket = None
    total_alerts = 0
    for _, row in df.iterrows():
        bucket = int(row["created_at"] // monitor.bucket_ms) * monitor.bucket_ms
        if current_bucket is not None and bucket != current_bucket:
            events = run_spike_check(monitor, alerter)
            total_alerts += len(events)
        current_bucket = bucket
        monitor.record(str(row["topic"]), float(row["created_at"]))

    # final check for the last bucket
    events = run_spike_check(monitor, alerter)
    total_alerts += len(events)

    print(f"\n=== Replay complete: {total_alerts} spike alert(s) fired ===")

    # cross-reference: which alert days coincide with a real 📢/🚨 incident broadcast?
    broadcasts = df[df["content"].astype(str).str.contains(r"📢|🚨", regex=True, na=False)]
    broadcast_days = {_fmt(t) for t in broadcasts["created_at"]}
    print(f"(days with 📢/🚨 incident broadcasts in the data: {sorted(broadcast_days)})")


if __name__ == "__main__":
    main()
