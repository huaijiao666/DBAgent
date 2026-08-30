"""Atomic, bounded persistence for one workspace-local DBA conversation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.trace import sanitize_for_storage

SESSION_VERSION = 1
MAX_SESSION_BYTES = 1_000_000


class SessionStore:
    """Store only the latest resumable session under the selected workspace."""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace.resolve() / ".forge" / "session.json"

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        if not self.path.is_file():
            raise ValueError("saved DBA session path is not a file")
        if self.path.stat().st_size > MAX_SESSION_BYTES:
            raise ValueError("saved DBA session exceeds the size limit")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("saved DBA session is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("saved DBA session must be an object")
        if value.get("version") != SESSION_VERSION:
            raise ValueError("saved DBA session has an unsupported version")
        return value

    def save(self, payload: Mapping[str, Any]) -> None:
        value = {
            "version": SESSION_VERSION,
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=".session-",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
