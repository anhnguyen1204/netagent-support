"""Alerter implementation backed by SMTP."""
from __future__ import annotations

import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from src.alerts.base import AlertRecord, Alerter


class EmailAlerter(Alerter):
    def __init__(self, smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str, to_address: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.to_address = to_address

    def send(self, alert: AlertRecord) -> None:
        if not self.smtp_host or not self.to_address:
            raise RuntimeError(
                "EmailAlerter is not configured (SMTP_HOST / ALERT_EMAIL_TO missing). "
                "Set them in the environment or use ALERTER_BACKEND=console."
            )
        ts = datetime.fromtimestamp(alert.triggered_at / 1000, tz=timezone.utc).astimezone()
        msg = EmailMessage()
        msg["Subject"] = f"[netAgent {alert.severity}] {alert.topic}"
        msg["From"] = self.smtp_user or "netagent@localhost"
        msg["To"] = self.to_address
        msg.set_content(f"{alert.message}\n\ntopic: {alert.topic}\ntime: {ts.isoformat()}")

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.smtp_user:
                server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
