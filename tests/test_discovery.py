from pathlib import Path

from forge.discovery import discover_workspace


def test_discovers_nearest_nested_project_root(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    package = project / "snake_game"
    package.mkdir(parents=True)
    (project / "README.md").write_text("demo", encoding="utf-8")
    (project / "tests").mkdir()

    result = discover_workspace(package)

    assert result.start == package.resolve()
    assert result.root == project.resolve()
    assert result.auto_detected is True
    assert result.markers == ("README.md", "tests")


def test_strong_marker_selects_current_directory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")

    result = discover_workspace(tmp_path)

    assert result.root == tmp_path.resolve()
    assert result.auto_detected is False
    assert "pyproject.toml" in result.markers
