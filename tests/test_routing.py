from dbagent.agent import TaskMode
from dbagent.agent.routing import TaskModeStore


def test_mode_store_accepts_one_semantic_mode_decision() -> None:
    store = TaskModeStore()

    result = store.apply({"mode": "code", "reason": "The user requests a fix."})

    assert result.success is True
    assert result.content["selected_mode"] == "code"
    assert store.decision is not None
    assert store.decision.mode is TaskMode.CODE


def test_mode_store_rejects_auto_or_a_later_mode_change() -> None:
    store = TaskModeStore()

    invalid = store.apply({"mode": "auto", "reason": "Undecided"})
    assert invalid.success is False
    assert store.decision is None

    assert store.apply({"mode": "ask", "reason": "Read-only request"}).success
    changed = store.apply({"mode": "code", "reason": "Changed my mind"})

    assert changed.success is False
    assert changed.content["error"] == "task mode cannot change after routing"
    assert store.decision is not None
    assert store.decision.mode is TaskMode.ASK
