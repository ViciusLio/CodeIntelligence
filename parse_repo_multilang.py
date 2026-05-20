"""
parse_repo_multilang.py — Multi-language repository parser for CodeIntelligence.

Discovers all files in a repository, delegates each file to the appropriate
parser plugin (Python, TypeScript, Go, YAML, Markdown, SQL), and writes a
JSONL file in the standard CodeIntelligence chunk format.

This is an ADDITION to parse_repo.py — it does NOT replace it.
parse_repo.py remains the canonical Python-only parser.

Usage:
    python parse_repo_multilang.py <repo_path>
    python parse_repo_multilang.py <repo_path> --output chunks.jsonl
    python parse_repo_multilang.py <repo_path> --lang python,typescript,go
    python parse_repo_multilang.py <repo_path> --lang all          (default)
    python parse_repo_multilang.py <repo_path> --exclude tests --exclude .venv

Output format (one JSON object per line, retrocompatible with parse_repo.py):
    {
        "id":       "<type>::<locator>",   # forward-slash always
        "type":     str,
        "language": str,                   # NEW field (optional for retrocompat)
        "text":     str,
        "metadata": dict
    }

The first chunk is always `repo_overview` listing all languages found.

Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Default excludes (mirrors parse_repo.py)
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDES = [
    ".venv", "venv", "env", ".env",
    "__pycache__", ".git", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", ".eggs",
]

_SUPPORTED_LANGS = ["python", "typescript", "go", "yaml", "markdown", "sql"]


# ---------------------------------------------------------------------------
# Repo overview builder (multilang variant)
# ---------------------------------------------------------------------------

def build_repo_overview(
    root: Path,
    lang_stats: dict[str, dict],  # {lang: {"files": int, "chunks": int}}
    exclude_patterns: list[str],
) -> dict:
    """Build the repo_overview chunk with multilang summary."""
    total_files = sum(v["files"] for v in lang_stats.values())
    total_chunks = sum(v["chunks"] for v in lang_stats.values())
    lang_summary_lines = [
        f"  {lang}: {v['files']} files -> {v['chunks']} chunks"
        for lang, v in sorted(lang_stats.items())
        if v["files"] > 0
    ]

    # Folder tree (top 2 levels)
    tree_lines: list[str] = []
    seen_dirs: set[str] = set()
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
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
                tree_lines.append(f"    {parts[1]}")
        if len(tree_lines) >= 40:
            break

    lines = [
        f"Repository root: {root.name}",
        f"Total files parsed: {total_files}",
        f"Total chunks: {total_chunks}",
        f"Languages detected:",
    ]
    lines.extend(lang_summary_lines or ["  (none)"])
    if tree_lines:
        lines.append("")
        lines.append("Folder layout (top 2 levels):")
        lines.extend(tree_lines[:40])

    return {
        "id": "repo_overview",
        "type": "repo_overview",
        "text": "\n".join(lines),
        "metadata": {
            "repo_name": root.name,
            "total_files": total_files,
            "total_chunks": total_chunks,
            "languages": {lang: v for lang, v in lang_stats.items() if v["files"] > 0},
            "excluded_patterns": exclude_patterns,
        },
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def iter_all_files(root: Path, exclude_patterns: list[str]):
    """Yield all files under root, skipping excluded paths."""
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        if any(pat in rel for pat in exclude_patterns):
            continue
        yield f


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_repo_multilang(
    repo_path: str | Path,
    output_path: str | Path = "repo_chunks_multilang.jsonl",
    extra_excludes: list[str] | None = None,
    lang_filter: list[str] | None = None,
) -> int:
    """
    Parse all files in repo_path using the registered parsers.

    Parameters
    ----------
    repo_path:
        Path to the repository root directory.
    output_path:
        Destination JSONL file.
    extra_excludes:
        Extra path patterns to exclude.
    lang_filter:
        List of language names to include (e.g. ["python", "typescript"]).
        Pass None or ["all"] to include all supported languages.

    Returns the total number of chunks written.
    """
    # Late import so the module can be imported without parsers on sys.path
    try:
        from parsers import parse_file as registry_parse_file  # type: ignore
        from parsers import list_parser_languages              # type: ignore
    except ImportError as exc:
        print(f"Error: cannot import parsers package: {exc}", file=sys.stderr)
        sys.exit(1)

    root = Path(repo_path).resolve()
    if not root.is_dir():
        print(f"Error: {repo_path!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    excludes = DEFAULT_EXCLUDES + (extra_excludes or [])
    use_filter = lang_filter and lang_filter != ["all"]
    active_langs = set(lang_filter) if use_filter else set(_SUPPORTED_LANGS)

    print(f"Scanning {root} ...")
    print(f"Languages: {', '.join(sorted(active_langs)) if use_filter else 'all'}")

    # Collect all chunks (deferred — we need lang_stats first for overview)
    all_chunks: list[dict] = []
    lang_stats: dict[str, dict] = defaultdict(lambda: {"files": 0, "chunks": 0})
    skipped = 0

    all_files = list(iter_all_files(root, excludes))
    print(f"  Found {len(all_files)} files after exclusions")

    for i, f in enumerate(all_files, 1):
        chunks = registry_parse_file(f, root)
        if not chunks:
            skipped += 1
            continue

        # Determine language from chunks (prefer the first chunk's language field)
        lang = chunks[0].get("language") if chunks else None
        if lang is None:
            # Derive from chunk type / id as fallback
            lang = "unknown"

        # Apply language filter
        if use_filter and lang not in active_langs:
            skipped += 1
            continue

        lang_stats[lang]["files"] += 1
        lang_stats[lang]["chunks"] += len(chunks)
        all_chunks.extend(chunks)

        if i % 50 == 0 or i == len(all_files):
            print(f"  Processed {i}/{len(all_files)} files - {len(all_chunks)} chunks ...")

    # Build overview (first chunk)
    overview = build_repo_overview(root, dict(lang_stats), excludes)

    # Write JSONL
    out = Path(output_path)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(overview, ensure_ascii=False) + "\n")
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total = 1 + len(all_chunks)  # 1 = overview
    print(f"\nDone.")
    print(f"  Skipped {skipped} files (no parser / filtered out)")
    print()
    print("Per-language summary:")
    for lang, stats in sorted(lang_stats.items()):
        if stats["files"] > 0:
            print(f"  {lang.capitalize()}: {stats['files']} files -> {stats['chunks']} chunks")
    print()
    print(f"Wrote {total} chunks -> {out}")
    return total


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    output = _pop_arg(args, "--output") or "repo_chunks_multilang.jsonl"
    excludes = _pop_multi(args, "--exclude")

    # --lang python,typescript,go  OR  --lang all
    lang_raw = _pop_arg(args, "--lang") or "all"
    if lang_raw == "all":
        lang_filter = None
    else:
        lang_filter = [l.strip() for l in lang_raw.split(",") if l.strip()]

    repo_path = args[0] if args else "."

    parse_repo_multilang(
        repo_path,
        output_path=output,
        extra_excludes=excludes if excludes else None,
        lang_filter=lang_filter,
    )


if __name__ == "__main__":
    main()
