"""Topic-frequency spike detector + alert trigger.

Maintains a rolling count of topic occurrences per time bucket; flags a spike when a
bucket's count exceeds mean + k*stddev of that topic's recent history, or a simple
absolute-count threshold during cold start (before enough history exists to compute a
stable baseline).

Bucket size defaults to 1 day: this dataset averages ~7 messages/hour but is bursty, and
per-topic counts are tiny at the hourly level (max ~3) — daily buckets are where real
incident waves (e.g. a "mất publish" day, a vmail-blocking day) actually show up as a
count clearly above the topic's normal daily volume.
"""
from __future__ import annotations

import math
from collections import defaultdict

from pydantic import BaseModel

from src.alerts.base import AlertRecord, Alerter

BUCKET_MS = 24 * 60 * 60 * 1000  # 1 day


class SpikeEvent(BaseModel):
    topic: str
    bucket_start: float  # unix epoch ms
    count: int
    baseline_mean: float
    baseline_stddev: float


class SpikeMonitor:
    def __init__(
        self,
        bucket_minutes: int = 24 * 60,
        k: float = 2.0,
        cold_start_abs_threshold: int = 4,
        min_history_buckets: int = 3,
        min_spike_count: int = 3,
    ):
        self.bucket_minutes = bucket_minutes
        self.bucket_ms = bucket_minutes * 60 * 1000
        self.k = k
        self.cold_start_abs_threshold = cold_start_abs_threshold
        self.min_history_buckets = min_history_buckets
        # a bucket with fewer than this many reports is never a spike, regardless of how
        # low the baseline is -- without this floor, a topic whose normal daily volume is
        # ~0 flags a single report as a "spike" (mean 0.1 + 2*0.4 stddev < 1), which is
        # just noise, not an incident wave.
        self.min_spike_count = min_spike_count
        # topic -> {bucket_start_ms: count}
        self._counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # topics we've already alerted on for a given bucket, to avoid duplicate alerts
        self._alerted: set[tuple[str, int]] = set()

    def _bucket_of(self, timestamp: float) -> int:
        return int(timestamp // self.bucket_ms) * self.bucket_ms

    def record(self, topic: str, timestamp: float) -> None:
        if topic in ("none", "other"):
            return
        bucket = self._bucket_of(timestamp)
        self._counts[topic][bucket] += 1

    def _baseline(self, topic: str, current_bucket: int) -> tuple[float, float, int]:
        """Mean/stddev of this topic's counts across all buckets strictly before the
        current one (empty buckets in between count as 0)."""
        buckets = self._counts[topic]
        prior = {b: c for b, c in buckets.items() if b < current_bucket}
        if not prior:
            return 0.0, 0.0, 0

        first = min(prior)
        n_buckets = int((current_bucket - first) // self.bucket_ms)
        if n_buckets <= 0:
            return 0.0, 0.0, 0

        # include zero-count buckets between first activity and now for a true baseline
        counts = [prior.get(first + i * self.bucket_ms, 0) for i in range(n_buckets)]
        mean = sum(counts) / len(counts)
        var = sum((c - mean) ** 2 for c in counts) / len(counts)
        return mean, math.sqrt(var), len(counts)

    def check_spikes(self) -> list[SpikeEvent]:
        events: list[SpikeEvent] = []
        for topic, buckets in self._counts.items():
            for bucket, count in buckets.items():
                if (topic, bucket) in self._alerted:
                    continue
                if count < self.min_spike_count:
                    continue  # too few reports to be a meaningful spike, whatever the baseline
                mean, stddev, n_history = self._baseline(topic, bucket)

                if n_history < self.min_history_buckets:
                    # cold start: not enough history for a stable baseline, use an
                    # absolute jump threshold instead
                    is_spike = count >= self.cold_start_abs_threshold
                else:
                    is_spike = count > mean + self.k * stddev and count > mean

                if is_spike:
                    self._alerted.add((topic, bucket))
                    events.append(
                        SpikeEvent(
                            topic=topic,
                            bucket_start=float(bucket),
                            count=count,
                            baseline_mean=mean,
                            baseline_stddev=stddev,
                        )
                    )
        events.sort(key=lambda e: e.bucket_start)
        return events


def run_spike_check(monitor: SpikeMonitor, alerter: Alerter) -> list[SpikeEvent]:
    events = monitor.check_spikes()
    for event in events:
        alerter.send(
            AlertRecord(
                topic=event.topic,
                message=(
                    f"Spike detected: {event.count} '{event.topic}' reports in one bucket "
                    f"(baseline mean={event.baseline_mean:.1f}, stddev={event.baseline_stddev:.1f})"
                ),
                severity="spike",
                triggered_at=event.bucket_start,
            )
        )
    return events
