from pathlib import Path

from forge.session_store import SessionStore


def test_session_store_is_atomic_bounded_and_redacts_secrets(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.save(
        {
            "conversation": [
                {"role": "user", "content": "key=sk-test-secret-value"}
            ],
            "session_context": {},
        }
    )
    loaded = store.load()

    assert loaded is not None
    assert loaded["version"] == 1
    assert loaded["conversation"][0]["content"] == "key=[REDACTED]"
    assert "sk-test-secret-value" not in store.path.read_text(encoding="utf-8")
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
