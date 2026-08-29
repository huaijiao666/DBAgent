from taskboard.models import Task
from taskboard.repository import InMemoryTaskRepository
from taskboard.service import TaskService


def make_service() -> TaskService:
    return TaskService(
        InMemoryTaskRepository(
            [
                Task(1, "Write docs", ("docs",)),
                Task(2, "Fix parser", ("bug", "backend")),
                Task(3, "Ship release", ("release",), completed=True),
            ]
        )
    )


def test_create_normalizes_title_and_tags() -> None:
    task = make_service().create("  Add tests  ", [" QA ", ""])

    assert task.task_id == 4
    assert task.title == "Add tests"
    assert task.tags == ("qa",)


def test_list_open_excludes_completed_tasks() -> None:
    assert [task.task_id for task in make_service().list_open()] == [1, 2]


def test_search_matches_title() -> None:
    assert [task.task_id for task in make_service().search("PARSER")] == [2]


def test_complete_marks_task_done() -> None:
    service = make_service()

    completed = service.complete(1)

    assert completed.completed is True
    assert service.list_open()[0].task_id == 2
