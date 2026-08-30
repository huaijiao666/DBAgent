"""Domain values for taskboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Task:
    """One task tracked by the service."""

    task_id: int
    title: str
    tags: tuple[str, ...] = ()
    completed: bool = False
