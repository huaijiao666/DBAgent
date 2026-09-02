import json
import shutil
from pathlib import Path

from dbagent.llm import FunctionCall
from dbagent.repository import (
    RelationKind,
    RepositoryIndex,
    RepositoryScanner,
    SymbolKind,
    extract_python_module,
)
from dbagent.repository.index import MAX_REPO_MAP_CHARACTERS
from dbagent.tools import create_readonly_registry
from dbagent.workspace import Workspace


def _copy_fixture(tmp_path: Path) -> Path:
    fixture = Path(__file__).parent / "fixtures" / "python_repository"
    workspace = tmp_path / "python_repository"
    shutil.copytree(fixture, workspace)
    return workspace


def _dispatch(registry, name: str, arguments: dict):
    return registry.dispatch(
        FunctionCall(
            call_id=f"call_{name}",
            name=name,
            arguments_json=json.dumps(arguments),
        )
    )


def test_repository_scanner_applies_builtin_and_gitignore_rules(
    tmp_path: Path,
) -> None:
    workspace_path = _copy_fixture(tmp_path)
    (workspace_path / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    generated = workspace_path / "generated"
    generated.mkdir()
    (generated / "auto.py").write_text("generated = True\n", encoding="utf-8")
    cache = workspace_path / "__pycache__"
    cache.mkdir()
    (cache / "cached.py").write_text("cached = True\n", encoding="utf-8")
    (workspace_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    scan = RepositoryScanner(Workspace(workspace_path)).scan()
    paths = [file.relative_path for file in scan.files]

    assert paths == sorted(paths)
    assert "README.md" in paths
    assert "web.js" in paths
    assert "src/acme/service.py" in paths
    assert "ignored.py" not in paths
    assert "generated/auto.py" not in paths
    assert "__pycache__/cached.py" not in paths
    assert ".env" not in paths
    assert scan.truncated is False


def test_python_ast_extracts_symbols_imports_and_locations(tmp_path: Path) -> None:
    workspace_path = _copy_fixture(tmp_path)
    source_path = workspace_path / "src" / "acme" / "models.py"

    module = extract_python_module(
        "src/acme/models.py", source_path.read_text(encoding="utf-8")
    )

    assert [item.compact() for item in module.imports] == [
        "dataclasses.{dataclass}"
    ]
    assert [(symbol.qualified_name, symbol.kind) for symbol in module.symbols] == [
        ("User", SymbolKind.CLASS),
        ("User.display_name", SymbolKind.METHOD),
    ]
    user, display_name = module.symbols
    assert user.line_start == 4
    assert user.signature == "User"
    assert display_name.signature == "display_name(self) -> str"
    assert display_name.docstring == "Return the user-facing name."
    assert display_name.line_end > display_name.line_start


def test_python_ast_reports_syntax_error_without_crashing_index() -> None:
    module = extract_python_module("broken.py", "def broken(:\n")

    assert module.symbols == ()
    assert module.imports == ()
    assert module.parse_error is not None
    assert "line 1" in module.parse_error


def test_symbol_index_builds_import_and_basic_symbol_relationships(
    tmp_path: Path,
) -> None:
    workspace_path = _copy_fixture(tmp_path)

    index = RepositoryIndex.build(Workspace(workspace_path))
    service = next(
        symbol for symbol in index.symbols if symbol.name == "UserService"
    )
    find = next(
        symbol
        for symbol in index.symbols
        if symbol.qualified_name == "UserService.find"
    )
    relations = index.relations_for(find.symbol_id)
    service_relations = index.relations_for(service.symbol_id)

    assert service.bases == ("BaseService",)
    assert any(
        relation.kind is RelationKind.INHERITS
        and relation.resolved
        and "BaseService" in relation.target
        for relation in service_relations
    )
    assert any(
        relation.kind is RelationKind.CONTAINS
        and relation.target == find.symbol_id
        for relation in service_relations
    )
    assert {
        relation.kind for relation in relations
    } == {RelationKind.CALLS}
    assert any("normalize_id" in relation.target for relation in relations)
    assert any("UserRepository.get" in relation.target for relation in relations)
    assert "web.js" not in {module.path for module in index.modules}


def test_compact_repository_map_contains_symbols_imports_and_relations(
    tmp_path: Path,
) -> None:
    workspace_path = _copy_fixture(tmp_path)

    repository_map = RepositoryIndex.build(Workspace(workspace_path)).render_map()

    assert repository_map.startswith("Python repository map: 5 files, 10 symbols")
    assert "src/acme/service.py" in repository_map
    assert "imports: .base.{BaseService}" in repository_map
    assert "class UserService(BaseService)" in repository_map
    assert "method find(self, raw_id: str) -> User | None" in repository_map
    assert "calls normalize_id" in repository_map
    assert "calls UserRepository.get" in repository_map
    assert "web.js" not in repository_map
    assert "ignored.py" not in repository_map


def test_repository_tools_search_and_read_only_the_target_symbol(
    tmp_path: Path,
) -> None:
    workspace_path = _copy_fixture(tmp_path)
    registry = create_readonly_registry(workspace_path)

    repository_map = _dispatch(registry, "get_repo_map", {})
    search = _dispatch(registry, "search_symbol", {"query": "UserService.find"})
    symbol_id = search.content["matches"][0]["symbol_id"]
    read = _dispatch(registry, "read_symbol", {"symbol_id": symbol_id})

    assert repository_map.success is True
    assert search.success is True
    assert search.content["match_count"] == 1
    assert search.content["truncated"] is False
    assert read.success is True
    assert read.content["symbol"]["qualified_name"] == "UserService.find"
    assert "10:     def find" in read.content["source"]
    assert "normalize_id(raw_id)" in read.content["source"]
    assert "class UserService" not in read.content["source"]
    assert "def __init__" not in read.content["source"]
    assert read.content["source_truncated"] is False


def test_repository_map_has_a_hard_character_budget(tmp_path: Path) -> None:
    source = tmp_path / "many_symbols.py"
    source.write_text(
        "\n\n".join(
            f"def function_{number}(argument: str) -> str:\n"
            f"    return argument + '{number}'"
            for number in range(1_000)
        )
        + "\n",
        encoding="utf-8",
    )

    repository_map = RepositoryIndex.build(Workspace(tmp_path)).render_map()

    assert len(repository_map) <= MAX_REPO_MAP_CHARACTERS
    assert repository_map.endswith("[repository map truncated]")
