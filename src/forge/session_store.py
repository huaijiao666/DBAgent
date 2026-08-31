"""Atomic, bounded persistence for workspace-local DBA conversations."""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.trace import sanitize_for_storage

SESSION_VERSION = 1
MAX_SESSION_BYTES = 1_000_000
_SESSION_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$")
_LEGACY_SESSION_ID = "legacy"


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Metadata sufficient to choose a session without loading its history."""

    session_id: str
    title: str
    updated_at: str
    turns: int
    status: str


class SessionStore:
    """Store multiple resumable sessions under the selected workspace.

    Each session has one independently replaceable JSON file. The old
    ``.forge/session.json`` format remains readable as the special ``legacy``
    session so upgrading DBA never silently discards the user's latest history.
    """

    def __init__(self, workspace: Path) -> None:
        forge_directory = workspace.resolve() / ".forge"
        self.sessions_directory = forge_directory / "sessions"
        self.legacy_path = forge_directory / "session.json"

    @property
    def path(self) -> Path:
        """Return the latest saved session path for backward compatibility."""

        summaries = self.list_sessions()
        if summaries:
            return self._path_for(summaries[0].session_id)
        return self.legacy_path

    @property
    def exists(self) -> bool:
        return bool(self.list_sessions())

    @staticmethod
    def new_session_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{secrets.token_hex(3)}"

    def list_sessions(self) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        if self.sessions_directory.is_dir():
            for path in self.sessions_directory.glob("*.json"):
                try:
                    value = self._load_path(path)
                    summaries.append(_summary(value, fallback_updated=path.stat().st_mtime))
                except (OSError, ValueError):
                    # One corrupt checkpoint must not hide other resumable work.
                    continue
        if self.legacy_path.is_file():
            try:
                value = self._load_path(self.legacy_path, legacy=True)
                summaries.append(
                    _summary(
                        value,
                        session_id=_LEGACY_SESSION_ID,
                        fallback_updated=self.legacy_path.stat().st_mtime,
                    )
                )
            except (OSError, ValueError):
                pass
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def load(self, session_id: str | None = None) -> dict[str, Any] | None:
        if session_id is None or session_id == "latest":
            summaries = self.list_sessions()
            if not summaries:
                return None
            session_id = summaries[0].session_id
        path = self._path_for(session_id)
        if not path.exists():
            return None
        return self._load_path(path, legacy=session_id == _LEGACY_SESSION_ID)

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        session_id: str | None = None,
    ) -> str:
        session_id = session_id or self.new_session_id()
        path = self._path_for(session_id)
        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        if path.is_file():
            try:
                existing = self._load_path(path)
                created_at = str(existing.get("created_at") or now)
            except (OSError, ValueError):
                pass
        value = {
            "version": SESSION_VERSION,
            "session_id": session_id,
            "created_at": created_at,
            "updated_at": now,
            **dict(payload),
        }
        serialized = json.dumps(
            sanitize_for_storage(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        encoded = serialized.encode("utf-8")
        if len(encoded) > MAX_SESSION_BYTES:
            raise ValueError("DBA session exceeds the size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{session_id}-",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return session_id

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            summaries = self.list_sessions()
            if not summaries:
                return
            session_id = summaries[0].session_id
        self._path_for(session_id).unlink(missing_ok=True)

    def _path_for(self, session_id: str) -> Path:
        if session_id == _LEGACY_SESSION_ID:
            return self.legacy_path
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("invalid DBA session id")
        return self.sessions_directory / f"{session_id}.json"

    def _load_path(self, path: Path, *, legacy: bool = False) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError("saved DBA session path is not a file")
        if path.stat().st_size > MAX_SESSION_BYTES:
            raise ValueError("saved DBA session exceeds the size limit")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("saved DBA session is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("saved DBA session must be an object")
        if value.get("version") != SESSION_VERSION:
            raise ValueError("saved DBA session has an unsupported version")
        if legacy:
            value = {**value, "session_id": _LEGACY_SESSION_ID}
        elif value.get("session_id") != path.stem:
            raise ValueError("saved DBA session id does not match its filename")
        return value


def _summary(
    value: Mapping[str, Any],
    *,
    session_id: str | None = None,
    fallback_updated: float,
) -> SessionSummary:
    conversation = value.get("conversation")
    messages = conversation if isinstance(conversation, list) else []
    turns = sum(
        isinstance(item, Mapping) and item.get("role") == "user"
        for item in messages
    )
    title = value.get("title")
    if not isinstance(title, str) or not title.strip():
        title = next(
            (
                str(item.get("content", ""))
                for item in messages
                if isinstance(item, Mapping) and item.get("role") == "user"
            ),
            "Untitled session",
        )
    context = value.get("session_context")
    status = "not_started"
    if isinstance(context, Mapping):
        status = str(context.get("verification_status") or "not_run")
    updated_at = value.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        updated_at = datetime.fromtimestamp(
            fallback_updated, tz=timezone.utc
        ).isoformat()
    return SessionSummary(
        session_id=session_id or str(value.get("session_id")),
        title=_single_line(title, 72),
        updated_at=updated_at,
        turns=int(turns),
        status=status,
    )


def _single_line(value: str, limit: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."
