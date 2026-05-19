"""
parse_repo.py — Parse a Python repository into semantic RAG chunks.

Generates repo_chunks.jsonl where each line is a self-contained chunk
ready for TF-IDF retrieval or embedding.

Usage:
    python parse_repo.py <repo_path>
    python parse_repo.py <repo_path> --output my_chunks.jsonl
    python parse_repo.py <repo_path> --exclude tests --exclude .venv

Chunk types produced:
    repo_overview              — folder structure, top-level packages, entry points
    file::<path>               — per .py file: imports, exports, module docstring
    function::<file>::<name>   — signature, docstring, full body, called functions
    class::<file>::<name>      — methods, attributes, inheritance

Output format (one JSON object per line):
    {
      "id":       "<type>::<locator>",
      "type":     "repo_overview" | "file" | "function" | "class",
      "text":     "<human-readable, embedding-ready prose>",
      "metadata": { ... type-specific structured fields ... }
    }

Requires: Python 3.9+ — zero external dependencies (ast, pathlib, json, re, sys).
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(path: Path, root: Path) -> str:
    """Return a forward-slash relative path string, e.g. 'app/core/security.py'."""
    return path.relative_to(root).as_posix()


def _clean_docstring(node: ast.AST) -> str:
    """Extract and dedent the docstring from a module/class/function node."""
    doc = ast.get_docstring(node, clean=True)
    return doc.strip() if doc else ""


def _collect_names_called(node: ast.AST) -> list[str]:
    """Return a deduplicated list of function/method names called inside a node."""
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return list(dict.fromkeys(names))  # preserve order, deduplicate


def _collect_imports(tree: ast.Module) -> tuple[list[str], list[str]]:
    """Return (stdlib_and_third_party_modules, from_imports) from a module AST."""
    modules: list[str] = []
    from_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                from_imports.append(f"{mod}.{alias.name}" if mod else alias.name)
    return list(dict.fromkeys(modules)), list(dict.fromkeys(from_imports))


def _top_level_names(tree: ast.Module) -> dict[str, list[str]]:
    """Return {'functions': [...], 'classes': [...], 'constants': [...]}."""
    functions, classes, constants = [], [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    constants.append(t.id)
    return {"functions": functions, "classes": classes, "constants": constants}


def _func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a readable function signature (no full source needed)."""
    args = node.args
    parts: list[str] = []

    # positional-only (Python 3.8+)
    for i, arg in enumerate(args.posonlyargs):
        parts.append(arg.arg)

    # regular args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        default_idx = i - defaults_offset
        if default_idx >= 0:
            try:
                default_src = ast.unparse(args.defaults[default_idx])
            except Exception:
                default_src = "..."
            parts.append(f"{arg.arg}={default_src}")
        else:
            parts.append(arg.arg)

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for i, arg in enumerate(args.kwonlyargs):
        kw_default = args.kw_defaults[i]
        if kw_default is not None:
            try:
                kw_src = ast.unparse(kw_default)
            except Exception:
                kw_src = "..."
            parts.append(f"{arg.arg}={kw_src}")
        else:
            parts.append(f"{arg.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = ""
    if node.returns:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return f"{prefix} {node.name}({', '.join(parts)}){ret}"


def _class_base_names(node: ast.ClassDef) -> list[str]:
    """Return list of base class names (best-effort, no evaluation)."""
    bases = []
    for b in node.bases:
        try:
            bases.append(ast.unparse(b))
        except Exception:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
    return bases


def _method_names(node: ast.ClassDef) -> list[str]:
    return [
        n.name
        for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _class_attributes(node: ast.ClassDef) -> list[str]:
    """Extract annotated class-level attributes (from __init__ and class body)."""
    attrs: list[str] = []
    for child in node.body:
        # class body annotations: x: int = ...
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            try:
                type_str = ast.unparse(child.annotation)
                attrs.append(f"{child.target.id}: {type_str}")
            except Exception:
                attrs.append(child.target.id)
        # __init__ self.x = ...
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
            for stmt in ast.walk(child):
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Attribute)
                    and isinstance(stmt.targets[0].value, ast.Name)
                    and stmt.targets[0].value.id == "self"
                ):
                    attrs.append(stmt.targets[0].attr)
    return list(dict.fromkeys(attrs))


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def build_file_chunk(path: Path, root: Path, tree: ast.Module, source: str) -> dict:
    slug = _slug(path, root)
    doc = _clean_docstring(tree)
    modules, from_imports = _collect_imports(tree)
    exports = _top_level_names(tree)

    lines = [
        f"File: {slug}",
    ]
    if doc:
        lines.append(f"Description: {doc}")
    if modules:
        lines.append(f"Imports: {', '.join(modules)}")
    if from_imports:
        lines.append(f"From-imports: {', '.join(from_imports[:20])}" +
                     (" …" if len(from_imports) > 20 else ""))
    if exports["functions"]:
        lines.append(f"Functions defined: {', '.join(exports['functions'])}")
    if exports["classes"]:
        lines.append(f"Classes defined: {', '.join(exports['classes'])}")
    if exports["constants"]:
        lines.append(f"Constants: {', '.join(exports['constants'])}")
    lines.append(f"Lines of code: {len(source.splitlines())}")

    return {
        "id": f"file::{slug}",
        "type": "file",
        "text": "\n".join(lines),
        "metadata": {
            "path": slug,
            "docstring": doc,
            "imports": modules,
            "from_imports": from_imports,
            "functions": exports["functions"],
            "classes": exports["classes"],
            "constants": exports["constants"],
            "loc": len(source.splitlines()),
        },
    }


def build_function_chunk(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    path: Path,
    root: Path,
    source_lines: list[str],
) -> dict:
    slug = _slug(path, root)
    name = func_node.name
    chunk_id = f"function::{slug}::{name}"

    signature = _func_signature(func_node)
    doc = _clean_docstring(func_node)
    called = _collect_names_called(func_node)

    # extract source body
    start = func_node.lineno - 1
    end = func_node.end_lineno
    body_lines = source_lines[start:end]
    body_src = textwrap.dedent("\n".join(body_lines))

    lines = [
        f"Function: {name}  (in {slug})",
        f"Signature: {signature}",
    ]
    if doc:
        lines.append(f"Docstring: {doc}")
    if called:
        lines.append(f"Calls: {', '.join(called[:30])}")
    lines.append("")
    lines.append("Source:")
    lines.append(body_src)

    return {
        "id": chunk_id,
        "type": "function",
        "text": "\n".join(lines),
        "metadata": {
            "file": slug,
            "name": name,
            "signature": signature,
            "docstring": doc,
            "calls": called,
            "start_line": func_node.lineno,
            "end_line": func_node.end_lineno,
            "is_async": isinstance(func_node, ast.AsyncFunctionDef),
        },
    }


def build_class_chunk(
    class_node: ast.ClassDef,
    path: Path,
    root: Path,
    source_lines: list[str],
) -> dict:
    slug = _slug(path, root)
    name = class_node.name
    chunk_id = f"class::{slug}::{name}"

    doc = _clean_docstring(class_node)
    bases = _class_base_names(class_node)
    methods = _method_names(class_node)
    attrs = _class_attributes(class_node)

    # extract source body
    start = class_node.lineno - 1
    end = class_node.end_lineno
    body_lines = source_lines[start:end]
    body_src = textwrap.dedent("\n".join(body_lines))
    # cap very long classes so chunks stay manageable
    if len(body_lines) > 200:
        body_src = "\n".join(body_lines[:200]) + "\n    # ... (truncated)"

    lines = [
        f"Class: {name}  (in {slug})",
    ]
    if bases:
        lines.append(f"Inherits from: {', '.join(bases)}")
    if doc:
        lines.append(f"Docstring: {doc}")
    if attrs:
        lines.append(f"Attributes: {', '.join(attrs[:20])}")
    if methods:
        lines.append(f"Methods: {', '.join(methods)}")
    lines.append("")
    lines.append("Source:")
    lines.append(body_src)

    return {
        "id": chunk_id,
        "type": "class",
        "text": "\n".join(lines),
        "metadata": {
            "file": slug,
            "name": name,
            "bases": bases,
            "docstring": doc,
            "attributes": attrs,
            "methods": methods,
            "start_line": class_node.lineno,
            "end_line": class_node.end_lineno,
        },
    }


# ---------------------------------------------------------------------------
# Repo overview builder
# ---------------------------------------------------------------------------

def build_repo_overview(
    root: Path,
    py_files: list[Path],
    exclude_patterns: list[str],
) -> dict:
    """Build the single repo_overview chunk."""

    # folder tree (top 2 levels)
    seen_dirs: set[str] = set()
    tree_lines: list[str] = []
    for f in sorted(py_files):
        rel = f.relative_to(root)
        parts = rel.parts
        if len(parts) >= 1:
            d = parts[0]
            if d not in seen_dirs:
                seen_dirs.add(d)
                tree_lines.append(f"  {d}/")
        if len(parts) >= 2:
            d2 = f"{parts[0]}/{parts[1]}"
            if d2 not in seen_dirs:
                seen_dirs.add(d2)
                # only add trailing slash for subdirectories, not .py files
                is_dir = not parts[1].endswith(".py")
                suffix = "/" if is_dir else ""
                tree_lines.append(f"    {parts[1]}{suffix}")

    # detect entry points (files with if __name__ == '__main__' or named main.py/cli.py/run.py)
    entry_points: list[str] = []
    for f in py_files:
        slug = f.relative_to(root).as_posix()
        name = f.name
        if name in ("main.py", "cli.py", "run.py", "__main__.py", "app.py"):
            entry_points.append(slug)
        else:
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
                if '__name__ == "__main__"' in src or "__name__ == '__main__'" in src:
                    entry_points.append(slug)
            except Exception:
                pass

    # top-level imports across the whole repo (for dependency detection)
    all_imports: dict[str, int] = {}
    for f in py_files:
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(f))
            mods, froms = _collect_imports(tree)
            for m in mods + [x.split(".")[0] for x in froms]:
                all_imports[m] = all_imports.get(m, 0) + 1
        except Exception:
            pass

    # top 20 most-imported modules, filter out same-repo relative imports (short names)
    top_deps = sorted(all_imports.items(), key=lambda x: -x[1])[:20]

    lines = [
        f"Repository root: {root.name}",
        f"Total Python files: {len(py_files)}",
        "",
        "Folder layout (top 2 levels):",
    ]
    lines.extend(tree_lines[:40])
    if entry_points:
        lines.append("")
        lines.append(f"Entry points: {', '.join(entry_points)}")
    lines.append("")
    lines.append("Most-imported modules (across repo):")
    for mod, count in top_deps:
        lines.append(f"  {mod} ({count} files)")

    return {
        "id": "repo_overview",
        "type": "repo_overview",
        "text": "\n".join(lines),
        "metadata": {
            "repo_name": root.name,
            "total_py_files": len(py_files),
            "entry_points": entry_points,
            "top_imports": {mod: count for mod, count in top_deps},
            "excluded_patterns": exclude_patterns,
        },
    }


# ---------------------------------------------------------------------------
# File walker
# ---------------------------------------------------------------------------

def iter_py_files(root: Path, exclude_patterns: list[str]) -> Iterator[Path]:
    """Yield all .py files under root, skipping excluded paths."""
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        skip = any(pat in rel for pat in exclude_patterns)
        if not skip:
            yield f


# ---------------------------------------------------------------------------
# Per-file chunk generator
# ---------------------------------------------------------------------------

def chunks_for_file(path: Path, root: Path) -> Iterator[dict]:
    """Parse one .py file and yield all its chunks (file + functions + classes)."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [skip] cannot read {path}: {exc}", file=sys.stderr)
        return

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        print(f"  [skip] syntax error in {path}: {exc}", file=sys.stderr)
        return

    source_lines = source.splitlines()

    # 1. file-level chunk
    yield build_file_chunk(path, root, tree, source)

    # 2. top-level functions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield build_function_chunk(node, path, root, source_lines)

    # 3. top-level classes (and their methods as separate function chunks)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield build_class_chunk(node, path, root, source_lines)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield build_function_chunk(child, path, root, source_lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDES = [
    ".venv", "venv", "env", ".env",
    "__pycache__", ".git", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", ".eggs",
]


def parse_repo(
    repo_path: str | Path,
    output_path: str | Path = "repo_chunks.jsonl",
    extra_excludes: list[str] | None = None,
) -> int:
    """
    Parse repo_path and write chunks to output_path.
    Returns the total number of chunks written.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        print(f"Error: {repo_path!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    excludes = DEFAULT_EXCLUDES + (extra_excludes or [])

    print(f"Scanning {root} …")
    py_files = list(iter_py_files(root, excludes))
    print(f"  Found {len(py_files)} Python files (after exclusions)")

    chunks: list[dict] = []

    # repo overview first
    chunks.append(build_repo_overview(root, py_files, excludes))

    # per-file chunks
    for i, f in enumerate(py_files, 1):
        file_chunks = list(chunks_for_file(f, root))
        chunks.extend(file_chunks)
        if i % 20 == 0 or i == len(py_files):
            print(f"  Parsed {i}/{len(py_files)} files — {len(chunks)} chunks so far …")

    # write JSONL
    out = Path(output_path)
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(chunks)} chunks -> {out}")
    return len(chunks)


def _pop_arg(args: list[str], flag: str, default=None):
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1] if idx + 1 < len(args) else default
        del args[idx: idx + 2]
        return val
    return default


def _pop_multi(args: list[str], flag: str) -> list[str]:
    """Pop all occurrences of --flag value from args list."""
    values = []
    while flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            values.append(args[idx + 1])
            del args[idx: idx + 2]
        else:
            del args[idx]
    return values


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    output = _pop_arg(args, "--output") or "repo_chunks.jsonl"
    excludes = _pop_multi(args, "--exclude")

    repo_path = args[0] if args else "."
    parse_repo(repo_path, output, excludes)


if __name__ == "__main__":
    main()
