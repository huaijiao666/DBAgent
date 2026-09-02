"""Thread-safe, local controls for one long-running agent task."""

from __future__ import annotations

import threading


class AgentRunControl:
    """Accept user steering and an abort request between local agent actions.

    The control object deliberately contains no model or tool logic.  The REPL
    (or another local UI) can feed it from a different thread, while
    :class:`AgentLoop` consumes it at deterministic boundaries: before a model
    request and before each local tool.  It cannot cancel an already executing
    HTTP request or subprocess, which is stated explicitly in the UI.
    """

    def __init__(self) -> None:
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self._abort_reason = "user requested abort"
        self._guidance: list[str] = []

    def request_abort(self, reason: str = "user requested abort") -> None:
        with self._lock:
            self._abort_reason = reason.strip() or "user requested abort"
        self._abort.set()

    @property
    def abort_requested(self) -> bool:
        return self._abort.is_set()

    @property
    def abort_reason(self) -> str:
        with self._lock:
            return self._abort_reason

    def submit_steering(self, message: str) -> bool:
        """Queue one non-empty user instruction for the next model turn."""

        normalized = message.strip()
        if not normalized:
            return False
        with self._lock:
            self._guidance.append(normalized)
        return True

    def drain_steering(self) -> tuple[str, ...]:
        """Return queued guidance exactly once, in user submission order."""

        with self._lock:
            messages = tuple(self._guidance)
            self._guidance.clear()
        return messages

    @property
    def pending_steering_count(self) -> int:
        """Return how many messages are waiting for the next model boundary."""

        with self._lock:
            return len(self._guidance)
