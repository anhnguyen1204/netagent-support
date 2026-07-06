"""FastAPI-based MessageSource stub.

Real version later: netChat (Mattermost) webhook posting into the same interface.
"""
from __future__ import annotations

from collections.abc import Iterator

from src.intake.base import IncomingMessage, MessageSource


class ApiIntake(MessageSource):
    """Receives messages pushed in-process from the FastAPI /ask endpoint.

    server.py constructs an IncomingMessage per request and feeds it directly to the
    agent graph; this class exists to satisfy the MessageSource interface uniformly
    with ReplayIntake.
    """

    def listen(self) -> Iterator[IncomingMessage]:
        raise NotImplementedError
