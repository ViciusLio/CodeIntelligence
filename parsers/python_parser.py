"""
parsers/python_parser.py — Python parser for the multi-language CodeIntelligence pipeline.

Wraps the existing parse_repo.py chunk builders so all logic stays in one place.
Adds "language": "python" to every chunk.

Interface:
    can_parse(path: Path) -> bool
    parse_file(path: Path, root: Path) -> list[dict]
"""

from __future__ import annotations

import sys
from pathlib import Path


def can_parse(path: Path) -> bool:
    """Return True for .py files."""
    return path.suffix == ".py"


def parse_file(path: Path, root: Path) -> list[dict]:
    """
    Parse a Python file by delegating to parse_repo.chunks_for_file().
    Injects "language": "python" into each chunk.
    Returns at least one fallback chunk on any error.
    """
    try:
        # Import parse_repo from the project root.  We use a lazy import so
        # the parsers package can be imported from anywhere without requiring
        # the project root on sys.path at module load time.
        import parse_repo  # type: ignore
    except ImportError:
        # Fallback: attempt to add the parent of parsers/ to sys.path
        parsers_dir = Path(__file__).parent
        project_root = parsers_dir.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        try:
            import parse_repo  # type: ignore  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                f"Cannot import parse_repo.py — make sure it is in the project root. ({exc})"
            ) from exc

    chunks = []
    try:
        for chunk in parse_repo.chunks_for_file(path, root):
            chunk = dict(chunk)
            chunk["language"] = "python"
            chunks.append(chunk)
    except Exception as exc:
        print(f"  [warn] python_parser error on {path}: {exc}", file=sys.stderr)
        # Graceful fallback
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:500]
        except Exception:
            content = ""
        slug = path.relative_to(root).as_posix()
        chunks = [{
            "id": f"file::{slug}",
            "type": "file",
            "language": "python",
            "text": f"File: {slug}\n\n[parse error: {exc}]\n\n{content}",
            "metadata": {"path": slug, "parse_error": str(exc)},
        }]

    return chunks
