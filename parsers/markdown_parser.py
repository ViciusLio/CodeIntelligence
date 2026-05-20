"""
parsers/markdown_parser.py — Markdown parser (stdlib only).

Splits Markdown into per-section chunks based on H1 / H2 / H3 headings.
Each section chunk includes the heading text + body up to the next heading.
Sections are truncated at 2000 characters.

Interface:
    can_parse(path: Path) -> bool
    parse_file(path: Path, root: Path) -> list[dict]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EXTENSIONS = {".md", ".mdx", ".markdown"}

# Matches H1, H2, H3 at line start
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)

_MAX_CHARS = 2000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_id_part(text: str) -> str:
    """Convert heading text to a safe ID fragment (lowercase, dashes)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\-. ]+", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80]


def _make_chunk(chunk_id: str, text: str, metadata: dict) -> dict:
    return {
        "id": chunk_id,
        "type": "doc",
        "language": "markdown",
        "text": text,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def can_parse(path: Path) -> bool:
    return path.suffix in _EXTENSIONS


def parse_file(path: Path, root: Path) -> list[dict]:
    """Parse a Markdown file into per-section chunks."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] markdown_parser cannot read {path}: {exc}", file=sys.stderr)
        slug = path.relative_to(root).as_posix()
        return [{
            "id": f"doc::{slug}::__root__",
            "type": "doc",
            "language": "markdown",
            "text": f"File: {slug}\n[read error: {exc}]",
            "metadata": {"path": slug, "parse_error": str(exc)},
        }]

    slug = _slug(path, root)
    matches = list(_HEADER_RE.finditer(source))

    if not matches:
        # No headings — emit single chunk with truncated content
        content = source[:_MAX_CHARS]
        if len(source) > _MAX_CHARS:
            content += "\n... (truncated)"
        return [_make_chunk(
            f"doc::{slug}::__root__",
            f"# {path.stem}\n\n{content}",
            {"path": slug, "heading": path.stem, "level": 0, "char_count": len(source)},
        )]

    chunks = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        section_start = m.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[section_start:section_end].strip()

        # Truncate
        if len(body) > _MAX_CHARS:
            body = body[:_MAX_CHARS] + "\n... (truncated)"

        heading_id = _safe_id_part(heading)
        cid = f"doc::{slug}::{heading_id}"

        chunks.append(_make_chunk(cid, body, {
            "path": slug,
            "heading": heading,
            "level": level,
            "char_count": len(body),
        }))

    return chunks
