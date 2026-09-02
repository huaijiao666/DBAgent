"""Model-facing repository map and symbol tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dbagent.llm import FunctionTool
from dbagent.repository import RepositoryIndex
from dbagent.repository.index import MAX_SYMBOL_SEARCH_RESULTS
from dbagent.tools.models import ToolDefinition, object_schema
from dbagent.tools.registry import ToolRegistry
from dbagent.workspace import Workspace

MAX_SYMBOL_SOURCE_CHARACTERS = 50_000


class RepositoryToolService:
    """Build a fresh lightweight index for each repository-aware request."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def get_repo_map(self) -> str:
        return RepositoryIndex.build(self._workspace).render_map()

    def search_symbol(self, query: str) -> dict[str, object]:
        index = RepositoryIndex.build(self._workspace)
        all_matches = index.search(query, limit=None)
        matches = all_matches[:MAX_SYMBOL_SEARCH_RESULTS]
        return {
            "query": query,
            "matches": [symbol.summary() for symbol in matches],
            "match_count": len(all_matches),
            "truncated": len(all_matches) > len(matches),
        }

    def read_symbol(self, symbol_id: str) -> dict[str, object]:
        index = RepositoryIndex.build(self._workspace)
        symbol = index.get(symbol_id)
        path = self._workspace.resolve(symbol.path)
        if not path.is_file():
            raise ValueError("symbol path is not a file")
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[symbol.line_start - 1 : symbol.line_end]
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=symbol.line_start)
        )
        truncated = len(numbered) > MAX_SYMBOL_SOURCE_CHARACTERS
        if truncated:
            numbered = (
                numbered[:MAX_SYMBOL_SOURCE_CHARACTERS]
                + f"\n[truncated after {MAX_SYMBOL_SOURCE_CHARACTERS} characters]"
            )
        return {
            "symbol": {
                **symbol.summary(),
                "parent": symbol.parent,
                "docstring": symbol.docstring,
            },
            "relationships": [
                {
                    "kind": relation.kind.value,
                    "target": relation.target,
                    "resolved": relation.resolved,
                }
                for relation in index.relations_for(symbol.symbol_id)
            ],
            "source": numbered,
            "source_truncated": truncated,
        }


def register_repository_tools(
    registry: ToolRegistry, workspace: Workspace
) -> None:
    """Add Python-aware context tools to an existing local registry."""

    service = RepositoryToolService(workspace)
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="get_repo_map",
                description=(
                    "Return a compact Python repository map with files, symbols, "
                    "imports, and basic resolved relationships."
                ),
                parameters=object_schema({}, required=[]),
            ),
            handler=lambda _arguments: service.get_repo_map(),
        )
    )
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="search_symbol",
                description=(
                    "Search indexed Python classes, functions, and methods by name, "
                    "qualified name, path, or symbol ID."
                ),
                parameters=object_schema(
                    {
                        "query": {
                            "type": "string",
                            "description": "Case-insensitive symbol search text.",
                        }
                    },
                    required=["query"],
                ),
            ),
            handler=lambda arguments: service.search_symbol(
                _required_string(arguments, "query")
            ),
        )
    )
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="read_symbol",
                description=(
                    "Read only one indexed Python class, function, or method and its "
                    "location metadata. Obtain symbol_id from search_symbol."
                ),
                parameters=object_schema(
                    {
                        "symbol_id": {
                            "type": "string",
                            "description": "Exact path::qualified_name@line symbol ID.",
                        }
                    },
                    required=["symbol_id"],
                ),
            ),
            handler=lambda arguments: service.read_symbol(
                _required_string(arguments, "symbol_id")
            ),
        )
    )


def _required_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
