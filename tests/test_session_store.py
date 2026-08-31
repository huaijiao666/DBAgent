from pathlib import Path

import pytest

from forge.session_store import SessionStore


def test_session_store_is_atomic_bounded_and_redacts_secrets(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.save(
        {
            "conversation": [
                {"role": "user", "content": "secret=synthetic-value"}
            ],
            "session_context": {},
        }
    )
    loaded = store.load()

    assert loaded is not None
    assert loaded["version"] == 1
    assert loaded["conversation"][0]["content"] == "[REDACTED]"
    assert "synthetic-value" not in store.path.read_text(encoding="utf-8")
    assert not list(store.path.parent.glob(".session-*.tmp"))

    store.save({"conversation": [], "session_context": {"turns": 0}})
    assert store.load()["session_context"]["turns"] == 0


def test_session_store_clear_removes_only_the_saved_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    unrelated = tmp_path / ".forge" / "trace.jsonl"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("trace\n", encoding="utf-8")
    store.save({"conversation": [], "session_context": {}})

    store.clear()

    assert store.exists is False
    assert unrelated.read_text(encoding="utf-8") == "trace\n"


def test_session_store_lists_and_loads_specific_sessions(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    first_id = store.save(
        {
            "title": "Fix parser",
            "conversation": [{"role": "user", "content": "Fix parser"}],
            "session_context": {"verification_status": "failed"},
        }
    )
    second_id = store.save(
        {
            "title": "Add CLI flag",
            "conversation": [{"role": "user", "content": "Add CLI flag"}],
            "session_context": {"verification_status": "passed"},
        }
    )

    summaries = store.list_sessions()

    assert {item.session_id for item in summaries} == {first_id, second_id}
    assert store.load(first_id)["title"] == "Fix parser"
    assert store.load(second_id)["session_context"]["verification_status"] == "passed"


def test_session_store_rejects_session_id_path_traversal(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    with pytest.raises(ValueError, match="invalid DBA session id"):
        store.load("../../outside")
