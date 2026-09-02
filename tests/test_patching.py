import os
import subprocess
from pathlib import Path

import dbagent.patching
import pytest
from dbagent.patching import PatchApplier
from dbagent.workspace import Workspace


def _hunk(old: list[str], new: list[str]) -> dict[str, list[str]]:
    return {"old_lines": old, "new_lines": new}


def test_multi_file_patch_is_applied_with_structured_result(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.py"
    first.write_bytes(b"alpha\r\nold\r\nomega\r\n")
    second.write_text("value = 1\n", encoding="utf-8")
    applier = PatchApplier(Workspace(tmp_path))

    result = applier.apply(
        [
            {
                "path": "first.txt",
                "hunks": [_hunk(["old"], ["new", "extra"])],
            },
            {
                "path": "second.py",
                "hunks": [_hunk(["value = 1"], ["value = 2"])],
            },
        ]
    )

    assert result["applied"] is True
    assert result["hunks_applied"] == 2
    assert result["failure_reason"] is None
    assert [item["path"] for item in result["changed_files"]] == [
        "first.txt",
        "second.py",
    ]
    assert all(
        item["before_sha256"] != item["after_sha256"]
        for item in result["changed_files"]
    )
    assert first.read_bytes() == b"alpha\r\nnew\r\nextra\r\nomega\r\n"
    assert second.read_text(encoding="utf-8") == "value = 2\n"


def test_context_mismatch_does_not_modify_any_file(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("before one\n", encoding="utf-8")
    second.write_text("before two\n", encoding="utf-8")
    applier = PatchApplier(Workspace(tmp_path))

    result = applier.apply(
        [
            {
                "path": "first.txt",
                "hunks": [_hunk(["before one"], ["after one"])],
            },
            {
                "path": "second.txt",
                "hunks": [_hunk(["missing context"], ["after two"])],
            },
        ]
    )

    assert result["applied"] is False
    assert result["changed_files"] == []
    assert result["hunks_applied"] == 0
    assert "context did not match" in result["failure_reason"]
    assert first.read_text(encoding="utf-8") == "before one\n"
    assert second.read_text(encoding="utf-8") == "before two\n"


def test_ambiguous_context_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "repeated.txt"
    target.write_text("same\nmiddle\nsame\n", encoding="utf-8")
    applier = PatchApplier(Workspace(tmp_path))

    result = applier.apply(
        [{"path": "repeated.txt", "hunks": [_hunk(["same"], ["changed"])]}]
    )

    assert result["applied"] is False
    assert "context is ambiguous (2 matches)" in result["failure_reason"]
    assert target.read_text(encoding="utf-8") == "same\nmiddle\nsame\n"


def test_patch_path_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    local = workspace / "local.txt"
    local.write_text("local\n", encoding="utf-8")
    applier = PatchApplier(Workspace(workspace))

    result = applier.apply(
        [
            {"path": "local.txt", "hunks": [_hunk(["local"], ["changed"])]},
            {
                "path": "../outside.txt",
                "hunks": [_hunk(["outside"], ["escaped"])],
            },
        ]
    )

    assert result["applied"] is False
    assert "path escapes workspace" in result["failure_reason"]
    assert local.read_text(encoding="utf-8") == "local\n"
    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_patch_target_through_external_directory_link_is_rejected(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "target.txt"
    target.write_text("outside\n", encoding="utf-8")
    link = workspace / "linked-directory"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"symlinks unavailable: {error}")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    applier = PatchApplier(Workspace(workspace))

    result = applier.apply(
        [
            {
                "path": "linked-directory/target.txt",
                "hunks": [_hunk(["outside"], ["escaped"])],
            }
        ]
    )

    assert result["applied"] is False
    assert "path escapes workspace" in result["failure_reason"]
    assert target.read_text(encoding="utf-8") == "outside\n"


def test_commit_failure_rolls_back_already_replaced_files(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first old\n", encoding="utf-8")
    second.write_text("second old\n", encoding="utf-8")
    real_replace = os.replace

    def fail_second_updated_file(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path == second
            and ".dbagent-updated-" in source_path.name
        ):
            raise OSError("injected second-file failure")
        real_replace(source, destination)

    monkeypatch.setattr(dbagent.patching.os, "replace", fail_second_updated_file)
    applier = PatchApplier(Workspace(tmp_path))

    result = applier.apply(
        [
            {
                "path": "first.txt",
                "hunks": [_hunk(["first old"], ["first new"])],
            },
            {
                "path": "second.txt",
                "hunks": [_hunk(["second old"], ["second new"])],
            },
        ]
    )

    assert result["applied"] is False
    assert "original files restored" in result["failure_reason"]
    assert first.read_text(encoding="utf-8") == "first old\n"
    assert second.read_text(encoding="utf-8") == "second old\n"
    assert not list(tmp_path.glob("*.dbagent-*.tmp"))


def test_sequential_hunks_use_the_in_memory_updated_content(tmp_path: Path) -> None:
    target = tmp_path / "steps.txt"
    target.write_text("one\ntwo\nthree", encoding="utf-8")
    applier = PatchApplier(Workspace(tmp_path))

    result = applier.apply(
        [
            {
                "path": "steps.txt",
                "hunks": [
                    _hunk(["one", "two"], ["one", "second"]),
                    _hunk(["second", "three"], ["second", "third"]),
                ],
            }
        ]
    )

    assert result["applied"] is True
    assert result["hunks_applied"] == 2
    assert target.read_text(encoding="utf-8") == "one\nsecond\nthird"
