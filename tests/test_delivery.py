from pathlib import Path

from forge.agent.delivery import DeliveryRequirements


def test_delivery_requirements_keep_explicit_relative_files_in_task_order(
    tmp_path: Path,
) -> None:
    requirements = DeliveryRequirements.from_task(
        "Create main.py, src/game.py, README.md, and test_game.py; run the tests after creation."
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "src" / "game.py").write_text("", encoding="utf-8")

    assert requirements.paths == (
        "main.py",
        "src/game.py",
        "README.md",
        "test_game.py",
    )
    assert requirements.missing(tmp_path) == ("README.md", "test_game.py")
    assert requirements.verification_kinds == ("test",)
    assert requirements.missing_verification_kinds({"compiler"}) == ("test",)


def test_delivery_requirements_ignore_workspace_escape_and_internal_paths() -> None:
    requirements = DeliveryRequirements.from_task(
        "Do not create ../secret.py or .forge/trace.jsonl; create app.py instead."
    )

    assert requirements.paths == ("app.py",)
