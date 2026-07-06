"""MessageSource interface — the Intake seam.

Stub today: REST endpoint / replay-from-CSV.
Real version later: netChat (Mattermost) webhook.
Do not import a concrete implementation directly from outside this package.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pydantic import BaseModel


class IncomingMessage(BaseModel):
    user_id: str
    content: str
    created_at: float  # unix epoch ms, matches raw CSV convention


class MessageSource(ABC):
    """Abstract source of incoming user messages/questions."""

    @abstractmethod
    def listen(self) -> Iterator[IncomingMessage]:
        """Yield messages as they arrive (live) or in order (replay)."""
        raise NotImplementedError
