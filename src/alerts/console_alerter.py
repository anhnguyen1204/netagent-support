"""Alerter implementation that prints to stdout/log. Default for local/demo runs."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.alerts.base import AlertRecord, Alerter

logger = logging.getLogger("netagent.alerts")


class ConsoleAlerter(Alerter):
    def send(self, alert: AlertRecord) -> None:
        ts = datetime.fromtimestamp(alert.triggered_at / 1000, tz=timezone.utc).astimezone()
        logger.warning(
            "[ALERT] severity=%s topic=%s at=%s | %s",
            alert.severity,
            alert.topic,
            ts.isoformat(),
            alert.message,
        )
        print(f"[ALERT] ({alert.severity}) topic={alert.topic} @ {ts.isoformat()}\n  {alert.message}")
