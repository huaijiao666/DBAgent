"""Explicit repository, symbol, import, and relationship values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SymbolKind(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class RelationKind(str, Enum):
    CONTAINS = "contains"
    INHERITS = "inherits"
    CALLS = "calls"


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    path: Path
    relative_path: str
    size: int


@dataclass(frozen=True, slots=True)
class RepositoryScan:
    files: tuple[RepositoryFile, ...]
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ImportInfo:
    module: str
    names: tuple[str, ...]
    line: int

    def compact(self) -> str:
        if not self.names:
            return self.module
        return f"{self.module}.{{{', '.join(self.names)}}}"


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    qualified_name: str
    kind: SymbolKind
    path: str
    line_start: int
    line_end: int
    signature: str
    parent: str | None = None
    docstring: str | None = None
    bases: tuple[str, ...] = field(default_factory=tuple)
    calls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def symbol_id(self) -> str:
        return f"{self.path}::{self.qualified_name}@{self.line_start}"

    def summary(self) -> dict[str, object]:
        return {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind.value,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class SymbolRelation:
    source_id: str
    kind: RelationKind
    target: str
    resolved: bool


@dataclass(frozen=True, slots=True)
class PythonModule:
    path: str
    imports: tuple[ImportInfo, ...]
    symbols: tuple[Symbol, ...]
    parse_error: str | None = None
