"""Command-line interface for the taskboard fixture."""

from __future__ import annotations

import argparse
import sys

from .service import TaskService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taskboard")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="create a task")
    add.add_argument("title")
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--priority", choices=["low", "normal", "high"], default="normal")

    list_command = commands.add_parser("list", help="list open tasks")
    list_command.add_argument(
        "--priority", choices=["low", "normal", "high"], default=None
    )

    done = commands.add_parser("done", help="complete a task")
    done.add_argument("task_id", type=int)

    search = commands.add_parser("search", help="search tasks")
    search.add_argument("query")
    return parser


def run(arguments: list[str], service: TaskService | None = None) -> str:
    parser = build_parser()
    options = parser.parse_args(arguments)
    task_service = service or TaskService()
    if options.command == "add":
        task = task_service.create(options.title, options.tag, options.priority)
        return _format_task(task)
    if options.command == "list":
        tasks = task_service.list_open()
        if options.priority:
            tasks = [task for task in tasks if task.priority == options.priority]
        return "\n".join(_format_task(task) for task in tasks)
    if options.command == "done":
        return _format_task(task_service.complete(options.task_id))
    return "\n".join(
        _format_task(task) for task in task_service.search(options.query)
    )


def _format_task(task: object) -> str:
    status = "done" if task.completed else "open"
    tags = ",".join(task.tags) if task.tags else "-"
    return f"{task.task_id}: {task.title} [{status}] tags={tags}"


if __name__ == "__main__":
    print(run(sys.argv[1:]))
