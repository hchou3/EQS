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

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# How long a session is kept before it's eligible for cleanup.
SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours


def _bundle_key(protected_attr: str, target_col: str) -> str:
    """Generate the shared dict key for a (protected_attr, target_col) pair."""
    return f"{protected_attr}::{target_col}"


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Pipeline artifacts — populated progressively as the user moves
    # through upload → bias analysis → train → shap.
    csv_data: Any = None              # CSVData instance
    fairlearn_dataset: Any = None     # FairlearnDataset, set after /train
    bundle_results: dict = field(default_factory=dict)  # key: _bundle_key -> BundleResult
    shap_results: dict = field(default_factory=dict)     # same key scheme -> ShapResult

    def touch(self) -> None:
        self.last_accessed = time.time()


class SessionStore:
    """Simple in-memory session registry with periodic expiration."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._cleanup_task = None
        self._start_cleanup_loop()

    def _start_cleanup_loop(self) -> None:
        """
        Launch a background asyncio task that calls purge_expired() every
        5 minutes. This prevents unbounded memory growth from abandoned
        sessions (e.g., uploads that were started but never followed up).
        """
        async def _loop():
            while True:
                await asyncio.sleep(300)  # 5 minutes
                removed = self.purge_expired()
                if removed > 0:
                    print(f"[session_store] purged {removed} expired session(s)")
        self._cleanup_task = asyncio.create_task(_loop())

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
        """
        Remove sessions past SESSION_TTL_SECONDS.
        Returns the number of sessions removed.
        """
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
