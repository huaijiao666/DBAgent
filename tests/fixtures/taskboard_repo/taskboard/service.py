"""Application operations for taskboard."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Task
from .repository import InMemoryTaskRepository


class TaskService:
    """Coordinate validation and persistence for task operations."""

    def __init__(self, repository: InMemoryTaskRepository | None = None) -> None:
        self.repository = repository or InMemoryTaskRepository()

    def create(self, title: str, tags: Iterable[str] = ()) -> Task:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("title must not be empty")
        clean_tags = tuple(tag.strip().casefold() for tag in tags if tag.strip())
        existing = self.repository.all()
        task = Task(
            task_id=max((item.task_id for item in existing), default=0) + 1,
            title=clean_title,
            tags=clean_tags,
        )
        self.repository.add(task)
        return task

    def list_open(self) -> list[Task]:
        return [task for task in self.repository.all() if not task.completed]

    def complete(self, task_id: int) -> Task:
        task = self.repository.get(task_id)
        updated = Task(
            task_id=task.task_id,
            title=task.title,
            tags=task.tags,
            completed=True,
        )
        self.repository.update(updated)
        return updated

    def search(self, query: str) -> list[Task]:
        normalized = query.strip().casefold()
        if not normalized:
            return []
        return [
            task
            for task in self.repository.all()
            if normalized in task.title.casefold()
        ]
