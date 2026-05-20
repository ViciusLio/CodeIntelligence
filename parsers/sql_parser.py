"""
parsers/sql_parser.py — SQL parser (regex-based, zero deps).

Extracts CREATE TABLE, CREATE VIEW, CREATE FUNCTION, CREATE PROCEDURE,
and CREATE INDEX statements (case-insensitive).  Each statement becomes
its own chunk.  Statements are truncated at 3000 characters.

Interface:
    can_parse(path: Path) -> bool
    parse_file(path: Path, root: Path) -> list[dict]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EXTENSIONS = {".sql"}

_MAX_CHARS = 3000

# Matches any CREATE [OR REPLACE] <object_type> <name>
# Captures: (object_type, object_name)
_CREATE_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:UNIQUE\s+)?"
    r"(TABLE|VIEW|FUNCTION|PROCEDURE|INDEX|TRIGGER|SEQUENCE|SCHEMA|TYPE|MATERIALIZED\s+VIEW)"
    r"\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:`([^`]+)`|\"([^\"]+)\"|(\w+(?:\.\w+)*))",
    re.IGNORECASE | re.DOTALL,
)

# Statement separator: semicolon possibly followed by whitespace / comments
_STMT_SPLIT = re.compile(r";[ \t]*(?:\n|$)", re.MULTILINE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _make_chunk(chunk_id: str, text: str, metadata: dict) -> dict:
    return {
        "id": chunk_id,
        "type": "sql_object",
        "language": "sql",
        "text": text,
        "metadata": metadata,
    }


def _split_statements(source: str) -> list[str]:
    """
    Split SQL source into individual statements on semicolons.
    Very simple (no full parser) — does not handle strings with semicolons.
    """
    parts = _STMT_SPLIT.split(source)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def can_parse(path: Path) -> bool:
    return path.suffix in _EXTENSIONS


def parse_file(path: Path, root: Path) -> list[dict]:
    """Parse a SQL file into per-CREATE-statement chunks."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] sql_parser cannot read {path}: {exc}", file=sys.stderr)
        slug = path.relative_to(root).as_posix()
        return [{
            "id": f"file::{slug}",
            "type": "file",
            "language": "sql",
            "text": f"File: {slug}\n[read error: {exc}]",
            "metadata": {"path": slug, "parse_error": str(exc)},
        }]

    slug = _slug(path, root)
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    statements = _split_statements(source)

    for stmt in statements:
        m = _CREATE_RE.search(stmt)
        if not m:
            continue

        obj_type = m.group(1).upper().replace("  ", " ")  # normalise whitespace
        # Name captured by one of three groups (backtick, double-quote, plain)
        obj_name = m.group(2) or m.group(3) or m.group(4) or "unknown"

        # Truncate long statements
        stmt_text = stmt
        if len(stmt_text) > _MAX_CHARS:
            stmt_text = stmt_text[:_MAX_CHARS] + "\n-- ... (truncated)"

        # Build unique ID (append counter if duplicate names)
        base_id = f"sql::{slug}::{obj_type}::{obj_name}"
        cid = base_id
        counter = 1
        while cid in seen_ids:
            cid = f"{base_id}_{counter}"
            counter += 1
        seen_ids.add(cid)

        text = (
            f"SQL {obj_type}: {obj_name}  (in {slug})\n"
            f"Language: SQL\n\n"
            f"Statement:\n{stmt_text}"
        )

        chunks.append(_make_chunk(cid, text, {
            "path": slug,
            "object_type": obj_type,
            "object_name": obj_name,
        }))

    if not chunks:
        # No CREATE statements found — emit a generic file chunk
        preview = source[:_MAX_CHARS]
        text = (
            f"SQL file: {slug}\n"
            f"Language: SQL\n"
            f"(No CREATE statements detected)\n\n"
            f"Content:\n{preview}"
        )
        return [_make_chunk(f"file::{slug}", text, {
            "path": slug,
            "object_type": None,
            "object_name": None,
        })]

    # File overview prepended
    obj_names = [c["metadata"]["object_name"] for c in chunks]
    file_text = (
        f"SQL file: {slug}\n"
        f"Language: SQL\n"
        f"Objects defined: {', '.join(obj_names)}\n"
        f"Total statements: {len(chunks)}"
    )
    file_chunk = {
        "id": f"file::{slug}",
        "type": "file",
        "language": "sql",
        "text": file_text,
        "metadata": {
            "path": slug,
            "objects": obj_names,
            "statement_count": len(chunks),
        },
    }
    return [file_chunk] + chunks
