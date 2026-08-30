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
    assert task.priority == "normal"


def test_create_accepts_valid_priority() -> None:
    task = make_service().create("Urgent fix", priority="high")

    assert task.priority == "high"


def test_create_rejects_invalid_priority() -> None:
    try:
        make_service().create("Unknown priority", priority="urgent")
    except ValueError as error:
        assert str(error) == "priority must be low, normal, or high"
    else:
        raise AssertionError("invalid priority was accepted")


def test_list_open_excludes_completed_tasks() -> None:
    assert [task.task_id for task in make_service().list_open()] == [1, 2]


def test_search_matches_title() -> None:
    assert [task.task_id for task in make_service().search("PARSER")] == [2]


def test_complete_marks_task_done() -> None:
    service = make_service()

    completed = service.complete(1)

    assert completed.completed is True
    assert service.list_open()[0].task_id == 2
