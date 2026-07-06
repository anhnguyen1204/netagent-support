"""Bước 3: load the Knowledge Base into the vector DB.

The KB is built from a hand-curated set of problem->solution pairs
(`data/curated_kb.csv`), NOT auto-extracted from conversation threads. Automated
extraction was tried (thread grouping -> resolution detection -> problem/solution
pairing) but this dataset yields almost no clean, resolved, single-topic exchanges: it
is mostly one-off messages and questions with no traceable reply. The whole
thread-extraction path was subsequently removed as dead code. See RESULTS.md Phase 4
for the full write-up. The curated set is grounded in real staff answers read directly
from the raw chat, so every KB entry is trustworthy.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.kb.schema import KBEntry
from src.kb.store import KBStore


def load_curated_kb(store: KBStore, curated_csv: Path) -> list[KBEntry]:
    """Load hand-curated problem->solution pairs from `curated_csv` into the KB.

    Each pair was read directly out of the raw chat and grounded in an actual staff
    answer (not invented). Marked with source_thread_id="curated" and confidence 0.95
    since each was hand-verified.
    """
    if not curated_csv.exists():
        return []
    df = pd.read_csv(curated_csv)
    entries = [
        KBEntry(
            problem=str(row["problem"]),
            solution=str(row["solution"]),
            topic=str(row["topic"]),
            source_thread_id="curated",
            confidence=0.95,
            created_at=0.0,
        )
        for _, row in df.iterrows()
    ]
    store.upsert(entries)
    return entries
