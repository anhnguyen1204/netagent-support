"""Bước 0: parse time, dedup, extract @mentions/entities.

Loads data/raw/output.csv, parses create_at (Unix epoch ms float -> datetime), dedups
exact (userId, content, create_at) triples, extracts @mentions and entities (workflow
URLs, IPs, node names) via regex. Writes data/processed/clean_messages.parquet.

The raw export has a handful of rows (~6 out of ~1500) where a message was truncated
mid UTF-8 character during export, breaking strict utf-8 decoding. We decode with
errors="replace" rather than failing the whole pipeline over a few corrupted trailing
characters in otherwise-usable messages.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

MENTION_RE = re.compile(r"@[\w.\-]+")
URL_RE = re.compile(r"https?://\S+")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Narrow node-name pattern: tokens explicitly prefixed/suffixed with a known
# infra-ish keyword, to avoid matching CSS/font/model names like "sans-serif".
NODE_RE = re.compile(
    r"\b(?:node|server|bot|workflow|service)-[\w\-]+\b", re.IGNORECASE
)


class CleanedMessage(BaseModel):
    user_id: str
    content: str
    created_at: float
    mentions: list[str]
    entities: list[str]


def load_raw(csv_path: Path) -> pd.DataFrame:
    raw_bytes = csv_path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    df = pd.read_csv(pd.io.common.StringIO(text))
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = df["create_at"].astype(float)
    df["created_at_dt"] = (
        pd.to_datetime(df["created_at"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Ho_Chi_Minh")
    )
    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["userId", "content", "create_at"], keep="first")


def extract_mentions(content: str) -> list[str]:
    if not isinstance(content, str):
        return []
    return MENTION_RE.findall(content)


def extract_entities(content: str) -> list[str]:
    if not isinstance(content, str):
        return []
    entities: list[str] = []
    entities.extend(URL_RE.findall(content))
    entities.extend(IP_RE.findall(content))
    entities.extend(NODE_RE.findall(content))
    return entities


def clean(csv_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_raw(csv_path)
    before_count = len(df)

    df = df.dropna(subset=["content"])
    df = parse_timestamps(df)
    df = dedup(df)

    df["mentions"] = df["content"].apply(extract_mentions)
    df["entities"] = df["content"].apply(extract_entities)

    df = df.rename(columns={"userId": "user_id"})
    out_cols = ["user_id", "content", "created_at", "created_at_dt", "mentions", "entities"]
    df = df[out_cols].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    after_count = len(df)
    with_mentions = (df["mentions"].str.len() > 0).sum()
    with_entities = (df["entities"].str.len() > 0).sum()
    print(f"rows before: {before_count}")
    print(f"rows after (dropna + dedup): {after_count}")
    print(f"rows with @mentions: {with_mentions}")
    print(f"rows with entities: {with_entities}")
    print(f"wrote: {output_path}")

    return df


if __name__ == "__main__":
    clean(Path("data/raw/output.csv"), Path("data/processed/clean_messages.parquet"))