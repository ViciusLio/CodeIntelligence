"""
parsers/go_parser.py — Go parser (regex-based, zero deps).

Extracts:
  - package declaration
  - import blocks (single and grouped)
  - func declarations (including methods with receivers)
  - struct type definitions: type Name struct { ... }
  - interface type definitions: type Name interface { ... }
  - top-level const / var blocks

Interface:
    can_parse(path: Path) -> bool
    parse_file(path: Path, root: Path) -> list[dict]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# package declaration
_PKG = re.compile(r"^package\s+(\w+)", re.MULTILINE)

# single import: import "pkg"
_IMPORT_SINGLE = re.compile(r'^import\s+"([^"]+)"', re.MULTILINE)

# grouped import block: import (\n  "a"\n  "b"\n)
_IMPORT_BLOCK = re.compile(r'import\s*\(([^)]*)\)', re.DOTALL)
_IMPORT_LINE = re.compile(r'"([^"]+)"')

# func with optional receiver:
#   func FuncName(        — top-level function
#   func (r RecvType) MethodName(  — method
#   func (r *RecvType) MethodName(
_FUNC_DECL = re.compile(
    r"^func\s+"
    r"(?:\(\s*\w+\s+\*?(\w+)\s*\)\s+)??"  # optional receiver: (recv RecvType)
    r"(\w+)\s*\(",
    re.MULTILINE,
)

# struct definition: type Foo struct {
_STRUCT_DECL = re.compile(r"^type\s+(\w+)\s+struct\s*\{", re.MULTILINE)

# interface definition: type Foo interface {
_IFACE_DECL = re.compile(r"^type\s+(\w+)\s+interface\s*\{", re.MULTILINE)

# const block: const ( ... ) or const Name = ...
_CONST_BLOCK = re.compile(r"^const\s+\(([^)]*)\)", re.MULTILINE | re.DOTALL)
_CONST_SINGLE = re.compile(r"^const\s+(\w+)\s*=", re.MULTILINE)

# var block: var ( ... ) or var Name type
_VAR_BLOCK = re.compile(r"^var\s+\(([^)]*)\)", re.MULTILINE | re.DOTALL)
_VAR_SINGLE = re.compile(r"^var\s+(\w+)\s+", re.MULTILINE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _extract_block(source: str, match_start: int, max_lines: int = 60) -> str:
    """Extract up to max_lines lines starting from the line containing match_start."""
    from_line = source[:match_start].count("\n")
    lines = source.splitlines()
    return "\n".join(lines[from_line: from_line + max_lines])


def _make_chunk(chunk_id: str, chunk_type: str, text: str, metadata: dict) -> dict:
    return {
        "id": chunk_id,
        "type": chunk_type,
        "language": "go",
        "text": text,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def can_parse(path: Path) -> bool:
    return path.suffix == ".go"


def parse_file(path: Path, root: Path) -> list[dict]:
    """Parse a Go source file into semantic chunks."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] go_parser cannot read {path}: {exc}", file=sys.stderr)
        slug = path.relative_to(root).as_posix()
        return [{
            "id": f"file::{slug}",
            "type": "file",
            "language": "go",
            "text": f"File: {slug}\n[read error: {exc}]",
            "metadata": {"path": slug, "parse_error": str(exc)},
        }]

    slug = _slug(path, root)
    chunks: list[dict] = []

    # ---- Package ----
    pkg_match = _PKG.search(source)
    package_name = pkg_match.group(1) if pkg_match else "unknown"

    # ---- Imports ----
    imports: list[str] = []
    for m in _IMPORT_SINGLE.finditer(source):
        imports.append(m.group(1))
    for block_m in _IMPORT_BLOCK.finditer(source):
        for imp_m in _IMPORT_LINE.finditer(block_m.group(1)):
            imports.append(imp_m.group(1))
    imports = list(dict.fromkeys(imports))

    # ---- Functions / Methods ----
    funcs: list[dict] = []
    for m in _FUNC_DECL.finditer(source):
        receiver_type = m.group(1)  # None for top-level funcs
        func_name = m.group(2)
        body = _extract_block(source, m.start(), max_lines=60)
        if receiver_type:
            display = f"Method: ({receiver_type}) {func_name}"
            cid = f"func::{slug}::({receiver_type}).{func_name}"
        else:
            display = f"Function: {func_name}"
            cid = f"func::{slug}::{func_name}"

        text = (
            f"{display}  (in {slug})\n"
            f"Package: {package_name}\n"
            f"Language: Go\n\n"
            f"Source:\n{body}"
        )
        chunk = _make_chunk(cid, "function", text, {
            "file": slug,
            "name": func_name,
            "receiver": receiver_type,
            "package": package_name,
        })
        chunks.append(chunk)
        funcs.append({"name": func_name, "receiver": receiver_type})

    # ---- Structs ----
    struct_names: list[str] = []
    for m in _STRUCT_DECL.finditer(source):
        name = m.group(1)
        body = _extract_block(source, m.start(), max_lines=40)
        cid = f"struct::{slug}::{name}"
        text = (
            f"Struct: {name}  (in {slug})\n"
            f"Package: {package_name}\n"
            f"Language: Go\n\n"
            f"Source:\n{body}"
        )
        chunks.append(_make_chunk(cid, "struct", text, {
            "file": slug, "name": name, "package": package_name,
        }))
        struct_names.append(name)

    # ---- Interfaces ----
    iface_names: list[str] = []
    for m in _IFACE_DECL.finditer(source):
        name = m.group(1)
        body = _extract_block(source, m.start(), max_lines=40)
        cid = f"interface::{slug}::{name}"
        text = (
            f"Interface: {name}  (in {slug})\n"
            f"Package: {package_name}\n"
            f"Language: Go\n\n"
            f"Source:\n{body}"
        )
        chunks.append(_make_chunk(cid, "interface", text, {
            "file": slug, "name": name, "package": package_name,
        }))
        iface_names.append(name)

    # ---- Constants ----
    const_names: list[str] = []
    for block_m in _CONST_BLOCK.finditer(source):
        # extract identifiers from block
        for line in block_m.group(1).splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                ident = re.match(r"(\w+)", line)
                if ident:
                    const_names.append(ident.group(1))
    for m in _CONST_SINGLE.finditer(source):
        const_names.append(m.group(1))
    const_names = list(dict.fromkeys(const_names))

    # ---- Var declarations ----
    var_names: list[str] = []
    for block_m in _VAR_BLOCK.finditer(source):
        for line in block_m.group(1).splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                ident = re.match(r"(\w+)", line)
                if ident:
                    var_names.append(ident.group(1))
    for m in _VAR_SINGLE.finditer(source):
        var_names.append(m.group(1))
    var_names = list(dict.fromkeys(var_names))

    # ---- File overview chunk ----
    func_names = [f["name"] for f in funcs]
    overview_lines = [
        f"File: {slug}",
        f"Language: Go",
        f"Package: {package_name}",
    ]
    if imports:
        overview_lines.append(f"Imports: {', '.join(imports[:20])}" +
                               (" ..." if len(imports) > 20 else ""))
    if func_names:
        overview_lines.append(f"Functions: {', '.join(func_names)}")
    if struct_names:
        overview_lines.append(f"Structs: {', '.join(struct_names)}")
    if iface_names:
        overview_lines.append(f"Interfaces: {', '.join(iface_names)}")
    if const_names:
        overview_lines.append(f"Constants: {', '.join(const_names[:20])}")
    if var_names:
        overview_lines.append(f"Variables: {', '.join(var_names[:20])}")
    overview_lines.append(f"Lines of code: {len(source.splitlines())}")

    file_chunk = _make_chunk(
        f"file::{slug}", "file", "\n".join(overview_lines),
        {
            "path": slug,
            "package": package_name,
            "imports": imports,
            "functions": func_names,
            "structs": struct_names,
            "interfaces": iface_names,
            "constants": const_names,
            "variables": var_names,
            "loc": len(source.splitlines()),
        }
    )
    chunks.insert(0, file_chunk)

    return chunks
