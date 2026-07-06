"""Unit tests for src/monitor/spike.py — bucketing, spike detection, alert firing."""
from __future__ import annotations

from tests.conftest import RecordingAlerter

from src.monitor.spike import SpikeMonitor, run_spike_check

DAY_MS = 24 * 60 * 60 * 1000


def _ts(day: int, n: int = 0) -> float:
    """timestamp on `day` (0-indexed days), nth message that day."""
    return float(day * DAY_MS + n * 1000)


def test_ignores_none_and_other_topics():
    m = SpikeMonitor()
    for i in range(10):
        m.record("none", _ts(0, i))
        m.record("other", _ts(0, i))
    assert m.check_spikes() == []


def test_min_spike_count_floor_blocks_tiny_counts():
    # 2 reports of a topic with ~zero baseline should NOT spike (the 43-alert bug):
    # min_spike_count defaults to 3.
    m = SpikeMonitor()
    m.record("credential", _ts(5, 0))
    m.record("credential", _ts(5, 1))
    assert m.check_spikes() == []


def test_cold_start_absolute_threshold_fires():
    # first-ever bucket with >= cold_start_abs_threshold (4) reports -> spike
    m = SpikeMonitor(cold_start_abs_threshold=4)
    for i in range(4):
        m.record("workflow_publish", _ts(0, i))
    events = m.check_spikes()
    assert len(events) == 1
    assert events[0].topic == "workflow_publish"
    assert events[0].count == 4


def test_statistical_spike_after_history():
    # quiet baseline of 1/day for several days, then a jump to 5 -> spike
    m = SpikeMonitor(min_spike_count=3, k=2.0)
    for day in range(6):
        m.record("email", _ts(day, 0))  # 1/day baseline
    for i in range(5):
        m.record("email", _ts(6, i))  # spike day
    events = m.check_spikes()
    spike_days = [int(e.bucket_start // DAY_MS) for e in events]
    assert 6 in spike_days


def test_no_duplicate_alerts_for_same_bucket():
    m = SpikeMonitor(cold_start_abs_threshold=4)
    for i in range(4):
        m.record("datatable", _ts(0, i))
    first = m.check_spikes()
    second = m.check_spikes()  # same state, already alerted
    assert len(first) == 1
    assert second == []


def test_run_spike_check_fires_alerter():
    alerter = RecordingAlerter()
    m = SpikeMonitor(cold_start_abs_threshold=4)
    for i in range(4):
        m.record("infra_incident", _ts(0, i))
    events = run_spike_check(m, alerter)
    assert len(events) == 1
    assert len(alerter.sent) == 1
    assert alerter.sent[0].severity == "spike"
    assert alerter.sent[0].topic == "infra_incident"
