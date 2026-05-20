"""
parsers/__init__.py — Parser registry and common interface for CodeIntelligence multi-language support.

Each parser module must expose two callables:
    can_parse(path: Path) -> bool
    parse_file(path: Path, root: Path) -> list[dict]

Chunk format (standard CodeIntelligence):
    {
        "id":       "<type>::<locator>",   # forward-slash always, even on Windows
        "type":     str,
        "language": str,                   # "python", "typescript", "go", "yaml", "markdown", "sql"
        "text":     str,                   # human-readable, embedding-ready
        "metadata": dict
    }

The "language" field is optional for retrocompatibility with legacy Python chunks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterator

# ---------------------------------------------------------------------------
# Lazy-import all parsers so missing optional deps never break the import
# ---------------------------------------------------------------------------

def _load_parsers():
    """Return the ordered list of (can_parse, parse_file) parser pairs."""
    parsers = []
    _parser_modules = [
        "parsers.python_parser",
        "parsers.typescript_parser",
        "parsers.go_parser",
        "parsers.yaml_parser",
        "parsers.markdown_parser",
        "parsers.sql_parser",
    ]
    for mod_name in _parser_modules:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            parsers.append((mod.can_parse, mod.parse_file, mod_name))
        except Exception as exc:
            print(f"  [warn] failed to load parser {mod_name}: {exc}", file=sys.stderr)
    return parsers


_PARSERS = None  # lazy init


def _get_parsers():
    global _PARSERS
    if _PARSERS is None:
        _PARSERS = _load_parsers()
    return _PARSERS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_parser(path: Path):
    """Return (can_parse_fn, parse_file_fn) for the first parser that claims the file, or None."""
    for can_parse, parse_file, _ in _get_parsers():
        try:
            if can_parse(path):
                return can_parse, parse_file
        except Exception:
            continue
    return None


def parse_file(path: Path, root: Path) -> list[dict]:
    """
    Parse a file using the appropriate registered parser.
    Returns an empty list if no parser handles this file type.
    On any exception: logs a warning and returns a minimal fallback chunk.
    """
    result = find_parser(path)
    if result is None:
        return []

    _, parser_fn = result
    try:
        return parser_fn(path, root)
    except Exception as exc:
        print(f"  [warn] parser error on {path}: {exc}", file=sys.stderr)
        # Best-effort fallback
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:500]
        except Exception:
            content = ""
        slug = path.relative_to(root).as_posix()
        return [{
            "id": f"file::{slug}",
            "type": "file",
            "text": f"File: {slug}\n\n[parse error: {exc}]\n\n{content}",
            "metadata": {"path": slug, "parse_error": str(exc)},
        }]


def iter_all_files(
    root: Path,
    exclude_patterns: list[str],
    lang_filter: list[str] | None = None,
) -> Iterator[Path]:
    """
    Yield all files under root that have at least one registered parser,
    skipping paths that match any exclude pattern.

    Parameters
    ----------
    root:
        Repository root directory.
    exclude_patterns:
        Substrings that, if found in the relative path, cause a file to be skipped.
    lang_filter:
        If provided, only files whose parser's language matches one of the listed
        language names are yielded.  Pass None or ["all"] to include everything.
    """
    # We need language info for filtering — build a language-aware check
    all_parsers = _get_parsers()
    use_filter = lang_filter and lang_filter != ["all"]

    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        if any(pat in rel for pat in exclude_patterns):
            continue

        for can_parse, parse_file_fn, mod_name in all_parsers:
            try:
                if can_parse(f):
                    if use_filter:
                        # derive language from module name, e.g. "parsers.python_parser" -> "python"
                        lang = mod_name.split(".")[-1].replace("_parser", "")
                        if lang not in lang_filter:
                            break  # skip this file (wrong language)
                    yield f
                    break
            except Exception:
                continue


def list_parser_languages() -> list[str]:
    """Return the list of language names supported by registered parsers."""
    langs = []
    for _, _, mod_name in _get_parsers():
        lang = mod_name.split(".")[-1].replace("_parser", "")
        langs.append(lang)
    return langs
