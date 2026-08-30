from taskboard.cli import run
from taskboard.models import Task
from taskboard.repository import InMemoryTaskRepository
from taskboard.service import TaskService


def test_cli_add_formats_tags() -> None:
    output = run(["add", "Write docs", "--tag", "docs"], TaskService())

    assert output == "1: Write docs [open] tags=docs"


def test_cli_add_sets_priority() -> None:
    service = TaskService()

    run(["add", "Fix parser", "--priority", "high"], service)

    assert service.repository.all()[0].priority == "high"


def test_cli_list_uses_service_state() -> None:
    service = TaskService(InMemoryTaskRepository([Task(2, "Fix parser")]))

    assert run(["list"], service) == "2: Fix parser [open] tags=-"


def test_cli_list_filters_open_tasks_by_priority() -> None:
    service = TaskService(
        InMemoryTaskRepository(
            [
                Task(1, "Low task", priority="low"),
                Task(2, "High task", priority="high"),
                Task(3, "Done high", completed=True, priority="high"),
            ]
        )
    )

    assert run(["list", "--priority", "high"], service) == (
        "2: High task [open] tags=-"
    )
