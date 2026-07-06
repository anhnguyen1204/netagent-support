"""Alerter interface — the Alerter seam.

Stub today: SMTP email / console log.
Real version later: netChat DM to KTV.
The system never claims to take an action — it answers and it alerts, nothing else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class AlertRecord(BaseModel):
    topic: str
    message: str
    severity: str
    triggered_at: float  # unix epoch ms


class Alerter(ABC):
    """Abstract sink for spike/escalation alerts."""

    @abstractmethod
    def send(self, alert: AlertRecord) -> None:
        raise NotImplementedError