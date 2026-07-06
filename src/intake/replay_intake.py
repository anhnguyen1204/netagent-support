"""Replays historical CSV messages through the system, time-ordered.

Used by scripts/replay_demo.py to validate the spike monitor against known incidents.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.intake.base import IncomingMessage, MessageSource


class ReplayIntake(MessageSource):
    def __init__(self, parquet_path: Path):
        self.parquet_path = parquet_path

    def listen(self) -> Iterator[IncomingMessage]:
        df = pd.read_parquet(self.parquet_path).sort_values("created_at")
        for _, row in df.iterrows():
            yield IncomingMessage(
                user_id=str(row["user_id"]),
                content=str(row["content"]),
                created_at=float(row["created_at"]),
            )
