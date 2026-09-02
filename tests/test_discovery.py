from pathlib import Path

from dbagent.discovery import discover_workspace, select_workspace


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


def test_default_selection_never_widens_exact_launch_directory(tmp_path: Path) -> None:
    parent = tmp_path / "test"
    launch = parent / "snake_game"
    launch.mkdir(parents=True)
    (parent / "pytest.ini").write_text("[pytest]", encoding="utf-8")
    (parent / "README.md").write_text("fixtures", encoding="utf-8")
    (parent / "tests").mkdir()

    result = select_workspace(launch)

    assert result.start == launch.resolve()
    assert result.root == launch.resolve()
    assert result.auto_detected is False


def test_parent_discovery_is_an_explicit_opt_in(tmp_path: Path) -> None:
    parent = tmp_path / "project"
    launch = parent / "src" / "package"
    launch.mkdir(parents=True)
    (parent / "pyproject.toml").write_text("[project]", encoding="utf-8")

    result = select_workspace(launch, discover_parent=True)

    assert result.root == parent.resolve()
    assert result.auto_detected is True
