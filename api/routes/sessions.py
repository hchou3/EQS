"""
In-memory session store.

Holds all per-upload state (CSVData, FairlearnDataset, BundleResults, etc.)
keyed by a session UUID. This is intentionally the *only* place that knows
how sessions are persisted — swapping this for Redis or a database later
should not require changes anywhere else in the API layer.

Not thread-safe beyond what Python's GIL gives you for free. Fine for a
single-process dev server; revisit if you move to multiple workers.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# How long a session is kept before it's eligible for cleanup.
SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Pipeline artifacts — populated progressively as the user moves
    # through upload → bias analysis → train → shap.
    csv_data: Any = None              # CSVData instance
    fairlearn_dataset: Any = None     # FairlearnDataset, set after /train
    bundle_results: dict = field(default_factory=dict)  # key: f"{protected_attr}::{target_col}" -> BundleResult
    shap_results: dict = field(default_factory=dict)     # same key scheme -> ShapResult

    def touch(self) -> None:
        self.last_accessed = time.time()


class SessionStore:
    """Simple in-memory session registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def purge_expired(self) -> int:
        """Remove sessions past SESSION_TTL_SECONDS. Returns count removed."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_accessed > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


# Module-level singleton — imported by route handlers.
session_store = SessionStore()