from taskboard.cli import run
from taskboard.models import Task
from taskboard.repository import InMemoryTaskRepository
from taskboard.service import TaskService


def test_cli_add_formats_tags() -> None:
    output = run(["add", "Write docs", "--tag", "docs"], TaskService())

    assert output == "1: Write docs [open] tags=docs"


def test_cli_list_uses_service_state() -> None:
    service = TaskService(InMemoryTaskRepository([Task(2, "Fix parser")]))

    assert run(["list"], service) == "2: Fix parser [open] tags=-"
