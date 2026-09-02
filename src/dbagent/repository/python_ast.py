"""Small, syntax-only Python symbol extractor built on the standard AST."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from dbagent.repository.models import ImportInfo, PythonModule, Symbol, SymbolKind


def extract_python_module(path: str, source: str) -> PythonModule:
    """Extract module imports and externally useful definitions."""

    try:
        tree = ast.parse(source, filename=path, type_comments=True)
    except SyntaxError as error:
        location = f"line {error.lineno}" if error.lineno else "unknown line"
        return PythonModule(
            path=path,
            imports=(),
            symbols=(),
            parse_error=f"{error.msg} ({location})",
        )

    imports = tuple(_extract_imports(tree.body))
    symbols: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(_function_symbol(path, node, parent=None))
        elif isinstance(node, ast.ClassDef):
            _append_class_symbols(path, node, parent=None, output=symbols)
    return PythonModule(path=path, imports=imports, symbols=tuple(symbols))


def _extract_imports(nodes: Iterable[ast.stmt]) -> Iterable[ImportInfo]:
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = (
                    alias.name
                    if alias.asname is None
                    else f"{alias.name} as {alias.asname}"
                )
                yield ImportInfo(module=name, names=(), line=node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            names = tuple(
                alias.name
                if alias.asname is None
                else f"{alias.name} as {alias.asname}"
                for alias in node.names
            )
            yield ImportInfo(module=module, names=names, line=node.lineno)


def _append_class_symbols(
    path: str,
    node: ast.ClassDef,
    *,
    parent: str | None,
    output: list[Symbol],
) -> None:
    qualified_name = node.name if parent is None else f"{parent}.{node.name}"
    output.append(
        Symbol(
            name=node.name,
            qualified_name=qualified_name,
            kind=SymbolKind.CLASS,
            path=path,
            line_start=_definition_start(node),
            line_end=node.end_lineno or node.lineno,
            signature=_class_signature(node),
            parent=parent,
            docstring=_docstring_summary(node),
            bases=tuple(_safe_unparse(base) for base in node.bases),
        )
    )
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            output.append(_function_symbol(path, child, parent=qualified_name))
        elif isinstance(child, ast.ClassDef):
            _append_class_symbols(
                path, child, parent=qualified_name, output=output
            )


def _function_symbol(
    path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    parent: str | None,
) -> Symbol:
    qualified_name = node.name if parent is None else f"{parent}.{node.name}"
    return Symbol(
        name=node.name,
        qualified_name=qualified_name,
        kind=SymbolKind.FUNCTION if parent is None else SymbolKind.METHOD,
        path=path,
        line_start=_definition_start(node),
        line_end=node.end_lineno or node.lineno,
        signature=_function_signature(node),
        parent=parent,
        docstring=_docstring_summary(node),
        calls=tuple(_collect_calls(node.body)),
    )


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    signature = f"{prefix}{node.name}({_safe_unparse(node.args)})"
    if node.returns is not None:
        signature += f" -> {_safe_unparse(node.returns)}"
    return signature


def _class_signature(node: ast.ClassDef) -> str:
    if not node.bases:
        return node.name
    return f"{node.name}({', '.join(_safe_unparse(base) for base in node.bases)})"


def _docstring_summary(node: ast.AST) -> str | None:
    docstring = ast.get_docstring(node, clean=True)
    if not docstring:
        return None
    first_line = docstring.splitlines()[0].strip()
    return first_line[:160]


def _definition_start(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    decorator_lines = [decorator.lineno for decorator in node.decorator_list]
    return min([node.lineno, *decorator_lines])


def _collect_calls(nodes: Iterable[ast.stmt]) -> Iterable[str]:
    collector = _CallCollector()
    for node in nodes:
        collector.visit(node)
    return collector.calls


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name and name not in self.calls:
            self.calls.append(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (TypeError, ValueError):
        return "?"
