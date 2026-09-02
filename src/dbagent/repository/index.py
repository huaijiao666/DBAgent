"""Python-first symbol index and compact repository map rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from dbagent.repository.models import (
    PythonModule,
    RelationKind,
    Symbol,
    SymbolRelation,
)
from dbagent.repository.python_ast import extract_python_module
from dbagent.repository.scanner import RepositoryScanner
from dbagent.workspace import Workspace

MAX_SYMBOL_SEARCH_RESULTS = 50
MAX_REPO_MAP_CHARACTERS = 20_000
MAX_PYTHON_SOURCE_BYTES = 2_000_000
MAX_DISPLAYED_IMPORTS_PER_FILE = 8
MAX_DISPLAYED_RELATIONS_PER_SYMBOL = 6


@dataclass(frozen=True, slots=True)
class RepositoryIndex:
    modules: tuple[PythonModule, ...]
    symbols: tuple[Symbol, ...]
    relations: tuple[SymbolRelation, ...]
    scan_truncated: bool = False

    @classmethod
    def build(cls, workspace: Workspace) -> RepositoryIndex:
        scan = RepositoryScanner(workspace).scan()
        modules: list[PythonModule] = []
        symbols: list[Symbol] = []
        for file in scan.files:
            if not file.relative_path.casefold().endswith(".py"):
                continue
            if file.size > MAX_PYTHON_SOURCE_BYTES:
                modules.append(
                    PythonModule(
                        path=file.relative_path,
                        imports=(),
                        symbols=(),
                        parse_error=(
                            f"file exceeds {MAX_PYTHON_SOURCE_BYTES} byte index limit"
                        ),
                    )
                )
                continue
            try:
                source = file.path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                modules.append(
                    PythonModule(
                        path=file.relative_path,
                        imports=(),
                        symbols=(),
                        parse_error="file is not valid UTF-8",
                    )
                )
                continue
            module = extract_python_module(file.relative_path, source)
            modules.append(module)
            symbols.extend(module.symbols)
        relations = _build_relations(symbols)
        return cls(
            modules=tuple(modules),
            symbols=tuple(symbols),
            relations=tuple(relations),
            scan_truncated=scan.truncated,
        )

    def search(
        self,
        query: str,
        *,
        limit: int | None = MAX_SYMBOL_SEARCH_RESULTS,
    ) -> tuple[Symbol, ...]:
        folded = query.strip().casefold()
        if not folded:
            raise ValueError("query must be a non-empty string")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive or None")

        def score(symbol: Symbol) -> tuple[int, str, int]:
            names = (
                symbol.name.casefold(),
                symbol.qualified_name.casefold(),
                symbol.symbol_id.casefold(),
            )
            if folded in names:
                rank = 0
            elif any(name.startswith(folded) for name in names):
                rank = 1
            elif any(folded in name for name in names):
                rank = 2
            else:
                rank = 3
            return rank, symbol.path, symbol.line_start

        matches = [
            symbol
            for symbol in self.symbols
            if folded in symbol.name.casefold()
            or folded in symbol.qualified_name.casefold()
            or folded in symbol.path.casefold()
            or folded in symbol.symbol_id.casefold()
        ]
        ordered = sorted(matches, key=score)
        return tuple(ordered if limit is None else ordered[:limit])

    def get(self, symbol_id: str) -> Symbol:
        for symbol in self.symbols:
            if symbol.symbol_id == symbol_id:
                return symbol
        raise KeyError(f"unknown symbol_id: {symbol_id}")

    def relations_for(self, symbol_id: str) -> tuple[SymbolRelation, ...]:
        return tuple(
            relation
            for relation in self.relations
            if relation.source_id == symbol_id
        )

    def render_map(self) -> str:
        relation_targets: dict[str, list[str]] = defaultdict(list)
        symbols_by_id = {symbol.symbol_id: symbol for symbol in self.symbols}
        for relation in self.relations:
            if relation.kind not in {RelationKind.INHERITS, RelationKind.CALLS}:
                continue
            target = symbols_by_id.get(relation.target)
            label = target.qualified_name if target else relation.target
            relation_targets[relation.source_id].append(
                f"{relation.kind.value} {label}"
            )

        header = (
            f"Python repository map: {len(self.modules)} files, "
            f"{len(self.symbols)} symbols"
        )
        lines = [header]
        character_count = len(header)

        def append(line: str) -> bool:
            nonlocal character_count
            added_characters = len(line) + 1
            if character_count + added_characters > MAX_REPO_MAP_CHARACTERS:
                return False
            lines.append(line)
            character_count += added_characters
            return True

        def truncated_map() -> str:
            nonlocal character_count
            marker = "[repository map truncated]"
            while (
                len(lines) > 1
                and character_count + len(marker) + 1
                > MAX_REPO_MAP_CHARACTERS
            ):
                removed = lines.pop()
                character_count -= len(removed) + 1
            lines.append(marker)
            return "\n".join(lines)

        if self.scan_truncated:
            if not append("[repository scan truncated]"):
                return truncated_map()
        for module in self.modules:
            if not append(module.path):
                return truncated_map()
            if module.parse_error:
                if not append(f"  [parse error: {module.parse_error}]"):
                    return truncated_map()
                continue
            if module.imports:
                imports = [
                    item.compact()
                    for item in module.imports[:MAX_DISPLAYED_IMPORTS_PER_FILE]
                ]
                suffix = (
                    ", ..."
                    if len(module.imports) > MAX_DISPLAYED_IMPORTS_PER_FILE
                    else ""
                )
                if not append(f"  imports: {', '.join(imports)}{suffix}"):
                    return truncated_map()
            if not module.symbols:
                if not append("  [no indexed symbols]"):
                    return truncated_map()
            for symbol in module.symbols:
                depth = symbol.qualified_name.count(".")
                indent = "  " * (depth + 1)
                line = (
                    f"{indent}{symbol.kind.value} {symbol.signature} "
                    f"[L{symbol.line_start}-{symbol.line_end}]"
                )
                relations = relation_targets.get(symbol.symbol_id, [])
                if relations:
                    shown = relations[:MAX_DISPLAYED_RELATIONS_PER_SYMBOL]
                    suffix = (
                        ", ..."
                        if len(relations) > MAX_DISPLAYED_RELATIONS_PER_SYMBOL
                        else ""
                    )
                    line += f"; {', '.join(shown)}{suffix}"
                if not append(line):
                    return truncated_map()
        return "\n".join(lines)


def _build_relations(symbols: list[Symbol]) -> list[SymbolRelation]:
    by_name: dict[str, list[Symbol]] = defaultdict(list)
    by_file_and_qualified: dict[tuple[str, str], Symbol] = {}
    for symbol in symbols:
        by_name[symbol.name].append(symbol)
        by_file_and_qualified[(symbol.path, symbol.qualified_name)] = symbol

    relations: list[SymbolRelation] = []
    for symbol in symbols:
        if symbol.parent:
            parent = by_file_and_qualified.get((symbol.path, symbol.parent))
            if parent:
                relations.append(
                    SymbolRelation(
                        source_id=parent.symbol_id,
                        kind=RelationKind.CONTAINS,
                        target=symbol.symbol_id,
                        resolved=True,
                    )
                )
        for base in symbol.bases:
            target = _resolve_unique_name(base, by_name)
            relations.append(
                SymbolRelation(
                    source_id=symbol.symbol_id,
                    kind=RelationKind.INHERITS,
                    target=target.symbol_id if target else base,
                    resolved=target is not None,
                )
            )
        for call in symbol.calls:
            target = _resolve_unique_name(call, by_name)
            if target and target.symbol_id != symbol.symbol_id:
                relations.append(
                    SymbolRelation(
                        source_id=symbol.symbol_id,
                        kind=RelationKind.CALLS,
                        target=target.symbol_id,
                        resolved=True,
                    )
                )
    return relations


def _resolve_unique_name(
    reference: str, by_name: dict[str, list[Symbol]]
) -> Symbol | None:
    simple_name = reference.rsplit(".", 1)[-1]
    candidates = by_name.get(simple_name, [])
    return candidates[0] if len(candidates) == 1 else None
