"""
parsers/typescript_parser.py — TypeScript / JavaScript parser (regex-based, zero deps).

Extracts:
  - function declarations: `function name(`, `const name = (`, `const name = async (`
  - class definitions: `class Name`
  - interface / type alias: `interface Name`, `type Name =`
  - export statements: export default, export const, export function, export class
  - A "file" overview chunk with imports + export summary

Interface:
    can_parse(path: Path) -> bool
    parse_file(path: Path, root: Path) -> list[dict]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Named function declarations: function foo( or export function foo(
_FUNC_DECL = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*[(<]",
    re.MULTILINE,
)

# Arrow / const functions: const foo = ( or const foo = async (
_ARROW_DECL = re.compile(
    r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(",
    re.MULTILINE,
)

# Class definitions
_CLASS_DECL = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+[\w, ]+)?",
    re.MULTILINE,
)

# Interface declarations
_INTERFACE_DECL = re.compile(
    r"^(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w, <>\[\]]+)?",
    re.MULTILINE,
)

# Type aliases: type Foo =
_TYPE_ALIAS = re.compile(
    r"^(?:export\s+)?type\s+(\w+)\s*(?:<[^>]*>)?\s*=",
    re.MULTILINE,
)

# Import statements: import ... from '...'
_IMPORT_FROM = re.compile(
    r"^import\s+(?:[^;]+?)\s+from\s+['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)

# require() calls: const x = require('...')
_REQUIRE = re.compile(r"require\(['\"]([^'\"]+)['\"]\)")

# Generic export detection (for overview)
_EXPORT_NAMED = re.compile(
    r"^export\s+(?:default\s+)?(?:const|let|var|function|class|interface|type|enum|async\s+function)\s+(\w+)",
    re.MULTILINE,
)
_EXPORT_DEFAULT = re.compile(r"^export\s+default\s+", re.MULTILINE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _extract_block(source: str, start_pos: int, max_lines: int = 60) -> str:
    """Extract lines from start_pos up to max_lines or the next top-level entity."""
    from_line = source[:start_pos].count("\n")
    all_lines = source.splitlines()
    block_lines = all_lines[from_line: from_line + max_lines]
    return "\n".join(block_lines)


def _make_chunk(chunk_id: str, chunk_type: str, text: str, metadata: dict) -> dict:
    return {
        "id": chunk_id,
        "type": chunk_type,
        "language": "typescript",
        "text": text,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def can_parse(path: Path) -> bool:
    return path.suffix in _EXTENSIONS


def parse_file(path: Path, root: Path) -> list[dict]:
    """Parse a TS/JS file into semantic chunks."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] typescript_parser cannot read {path}: {exc}", file=sys.stderr)
        slug = path.relative_to(root).as_posix()
        return [{
            "id": f"file::{slug}",
            "type": "file",
            "language": "typescript",
            "text": f"File: {slug}\n[read error: {exc}]",
            "metadata": {"path": slug, "parse_error": str(exc)},
        }]

    slug = _slug(path, root)
    chunks: list[dict] = []

    # ---- Collect entity names (deduplicated) ----
    funcs = list(dict.fromkeys(
        m.group(1) for m in _FUNC_DECL.finditer(source)
    ))
    arrow_funcs = list(dict.fromkeys(
        m.group(1) for m in _ARROW_DECL.finditer(source)
    ))
    classes = list(dict.fromkeys(
        (m.group(1), m.group(2)) for m in _CLASS_DECL.finditer(source)
    ))
    interfaces = list(dict.fromkeys(
        m.group(1) for m in _INTERFACE_DECL.finditer(source)
    ))
    type_aliases = list(dict.fromkeys(
        m.group(1) for m in _TYPE_ALIAS.finditer(source)
    ))
    imports = list(dict.fromkeys(
        m.group(1) for m in _IMPORT_FROM.finditer(source)
    )) + list(dict.fromkeys(
        m.group(1) for m in _REQUIRE.finditer(source)
    ))
    imports = list(dict.fromkeys(imports))
    exports = list(dict.fromkeys(
        m.group(1) for m in _EXPORT_NAMED.finditer(source)
    ))
    has_default_export = bool(_EXPORT_DEFAULT.search(source))

    # ---- Per-function chunks ----
    for m in _FUNC_DECL.finditer(source):
        name = m.group(1)
        body = _extract_block(source, m.start(), max_lines=50)
        text = (
            f"Function: {name}  (in {slug})\n"
            f"Language: TypeScript\n\n"
            f"Source:\n{body}"
        )
        cid = f"function::{slug}::{name}"
        chunks.append(_make_chunk(cid, "function", text, {
            "file": slug, "name": name, "kind": "function",
        }))

    # ---- Per-arrow-function chunks ----
    for m in _ARROW_DECL.finditer(source):
        name = m.group(1)
        body = _extract_block(source, m.start(), max_lines=50)
        text = (
            f"Function: {name}  (in {slug})\n"
            f"Language: TypeScript\n"
            f"Kind: arrow/const\n\n"
            f"Source:\n{body}"
        )
        cid = f"function::{slug}::{name}"
        # avoid duplicates with function declarations
        if not any(c["id"] == cid for c in chunks):
            chunks.append(_make_chunk(cid, "function", text, {
                "file": slug, "name": name, "kind": "arrow",
            }))

    # ---- Per-class chunks ----
    for m in _CLASS_DECL.finditer(source):
        name, base = m.group(1), m.group(2)
        body = _extract_block(source, m.start(), max_lines=80)
        base_str = f" extends {base}" if base else ""
        text = (
            f"Class: {name}{base_str}  (in {slug})\n"
            f"Language: TypeScript\n\n"
            f"Source:\n{body}"
        )
        cid = f"class::{slug}::{name}"
        chunks.append(_make_chunk(cid, "class", text, {
            "file": slug, "name": name, "base": base,
        }))

    # ---- Per-interface chunks ----
    for m in _INTERFACE_DECL.finditer(source):
        name = m.group(1)
        body = _extract_block(source, m.start(), max_lines=40)
        text = (
            f"Interface: {name}  (in {slug})\n"
            f"Language: TypeScript\n\n"
            f"Source:\n{body}"
        )
        cid = f"interface::{slug}::{name}"
        chunks.append(_make_chunk(cid, "interface", text, {
            "file": slug, "name": name,
        }))

    # ---- Per-type-alias chunks ----
    for m in _TYPE_ALIAS.finditer(source):
        name = m.group(1)
        body = _extract_block(source, m.start(), max_lines=20)
        text = (
            f"Type alias: {name}  (in {slug})\n"
            f"Language: TypeScript\n\n"
            f"Source:\n{body}"
        )
        cid = f"type::{slug}::{name}"
        chunks.append(_make_chunk(cid, "type", text, {
            "file": slug, "name": name,
        }))

    # ---- File overview chunk ----
    all_func_names = list(dict.fromkeys(funcs + arrow_funcs))
    class_names = [name for name, _ in classes]
    overview_lines = [f"File: {slug}", "Language: TypeScript"]
    if imports:
        overview_lines.append(f"Imports: {', '.join(imports[:20])}" +
                               (" ..." if len(imports) > 20 else ""))
    if exports:
        overview_lines.append(f"Exports: {', '.join(exports[:20])}" +
                               (" ..." if len(exports) > 20 else ""))
    if has_default_export:
        overview_lines.append("Has default export: yes")
    if all_func_names:
        overview_lines.append(f"Functions: {', '.join(all_func_names)}")
    if class_names:
        overview_lines.append(f"Classes: {', '.join(class_names)}")
    if interfaces:
        overview_lines.append(f"Interfaces: {', '.join(interfaces)}")
    if type_aliases:
        overview_lines.append(f"Type aliases: {', '.join(type_aliases)}")
    overview_lines.append(f"Lines of code: {len(source.splitlines())}")

    file_chunk = _make_chunk(
        f"file::{slug}", "file", "\n".join(overview_lines),
        {
            "path": slug,
            "imports": imports,
            "exports": exports,
            "functions": all_func_names,
            "classes": class_names,
            "interfaces": interfaces,
            "type_aliases": type_aliases,
            "has_default_export": has_default_export,
            "loc": len(source.splitlines()),
        }
    )
    # File chunk goes first
    chunks.insert(0, file_chunk)

    return chunks
