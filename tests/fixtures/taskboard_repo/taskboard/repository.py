"""In-memory persistence used by the taskboard service."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Task


class InMemoryTaskRepository:
    """Store tasks by id while keeping insertion order for iteration."""

    def __init__(self, tasks: Iterable[Task] = ()) -> None:
        self._tasks: dict[int, Task] = {task.task_id: task for task in tasks}

    def add(self, task: Task) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"task id already exists: {task.task_id}")
        self._tasks[task.task_id] = task

    def get(self, task_id: int) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise KeyError(f"unknown task id: {task_id}") from error

    def update(self, task: Task) -> None:
        if task.task_id not in self._tasks:
            raise KeyError(f"unknown task id: {task.task_id}")
        self._tasks[task.task_id] = task

    def all(self) -> list[Task]:
        return list(self._tasks.values())
