"""Unit tests for config loading, gate thresholds, and alerter implementations."""
from __future__ import annotations

import time

import pytest

from src.alerts.base import AlertRecord
from src.alerts.console_alerter import ConsoleAlerter
from src.alerts.email_alerter import EmailAlerter
from src.config import GateThresholds, load_settings
from src.llm.null_llm import NullLLM


def test_load_settings_defaults(monkeypatch):
    for var in ["LLM_BACKEND", "QDRANT_PORT", "ALERTER_BACKEND"]:
        monkeypatch.delenv(var, raising=False)
    s = load_settings()
    assert s.llm_backend == "null"
    assert s.qdrant_port == 6333
    assert s.alerter_backend == "console"
    assert isinstance(s.gate, GateThresholds)


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("QDRANT_PORT", "7000")
    s = load_settings()
    assert s.llm_backend == "ollama"
    assert s.qdrant_port == 7000


def test_gate_thresholds_ordering():
    g = GateThresholds()
    assert g.auto_reply_min > g.suggest_to_staff_min


def test_null_llm_complete_raises():
    # NullLLM must not silently no-op -- callers rely on the error to use their fallback
    with pytest.raises(RuntimeError):
        NullLLM().complete("anything")


def test_console_alerter_prints(capsys):
    ConsoleAlerter().send(
        AlertRecord(topic="credential", message="test alert", severity="spike", triggered_at=time.time() * 1000)
    )
    out = capsys.readouterr().out
    assert "credential" in out
    assert "test alert" in out


def test_email_alerter_unconfigured_raises():
    # not configured (no SMTP_HOST) -> clear error, not a silent failure
    alerter = EmailAlerter(smtp_host="", smtp_port=587, smtp_user="", smtp_password="", to_address="")
    with pytest.raises(RuntimeError):
        alerter.send(
            AlertRecord(topic="x", message="m", severity="spike", triggered_at=0.0)
        )
