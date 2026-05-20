"""
parsers/yaml_parser.py — YAML parser (stdlib only, NO import yaml).

Parses YAML files line-by-line for structured content:
  - Docker Compose: each service under `services:` becomes its own chunk
  - Kubernetes: documents separated by `---` with `kind:` and `metadata.name:` become chunks
  - GitHub Actions: each job under `jobs:` becomes its own chunk
  - Fallback: a single chunk with the raw content for generic YAML

Interface:
    can_parse(path: Path) -> bool
    parse_file(path: Path, root: Path) -> list[dict]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

_EXTENSIONS = {".yml", ".yaml"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _make_chunk(chunk_id: str, chunk_type: str, text: str, metadata: dict) -> dict:
    return {
        "id": chunk_id,
        "type": chunk_type,
        "language": "yaml",
        "text": text,
        "metadata": metadata,
    }


def _indent_level(line: str) -> int:
    """Return the number of leading spaces in a line."""
    return len(line) - len(line.lstrip(" "))


def _get_top_level_keys(lines: list[str]) -> list[str]:
    """Return key names at indentation 0."""
    keys = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_\-]*):\s*", line)
        if m:
            keys.append(m.group(1))
    return keys


def _extract_children_block(lines: list[str], parent_key: str, child_indent: int = 2) -> dict[str, str]:
    """
    Given lines of YAML, find all direct children of parent_key at child_indent.
    Returns dict mapping child_name -> raw block text.
    """
    children: dict[str, list[str]] = {}
    current_child: Optional[str] = None
    inside_parent = False

    for line in lines:
        # skip comments and empty lines
        stripped = line.strip()
        indent = _indent_level(line)

        # detect parent key at indent 0
        if indent == 0:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_\-]*):\s*", line)
            if m:
                if m.group(1) == parent_key:
                    inside_parent = True
                    current_child = None
                else:
                    inside_parent = False
                    current_child = None
            continue

        if not inside_parent:
            continue

        # child keys at child_indent
        if indent == child_indent:
            m = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_\-]*):\s*", line)
            if m:
                current_child = m.group(2)
                children[current_child] = [line]
                continue

        # content belonging to current child (deeper indent)
        if current_child is not None and indent > child_indent:
            children[current_child].append(line)

    return {k: "\n".join(v) for k, v in children.items()}


def _detect_yaml_type(lines: list[str], top_keys: list[str]) -> str:
    """Heuristic: detect docker-compose, k8s, github-actions, or generic."""
    top_set = set(top_keys)

    # Docker Compose
    if "services" in top_set and ("version" in top_set or "networks" in top_set or "volumes" in top_set):
        return "docker-compose"
    if "services" in top_set:
        return "docker-compose"

    # GitHub Actions workflow
    if "jobs" in top_set and ("on" in top_set or "name" in top_set):
        return "github-actions"
    if "jobs" in top_set:
        return "github-actions"

    # Kubernetes: look for `kind:` at top level
    for line in lines:
        m = re.match(r"^kind:\s+(\w+)", line)
        if m:
            return "kubernetes"

    return "generic"


def _parse_k8s_documents(source: str, slug: str) -> list[dict]:
    """Split on `---` separators and parse each k8s document."""
    docs = source.split("\n---")
    chunks = []
    for i, doc in enumerate(docs):
        doc = doc.strip()
        if not doc:
            continue
        doc_lines = doc.splitlines()

        # Extract kind and name
        kind = None
        name = None
        in_metadata = False
        for line in doc_lines:
            m_kind = re.match(r"^kind:\s+(\w+)", line)
            if m_kind:
                kind = m_kind.group(1)
            if re.match(r"^metadata:", line):
                in_metadata = True
            if in_metadata:
                m_name = re.match(r"^\s+name:\s+(.+)", line)
                if m_name:
                    name = m_name.group(1).strip()
                    break
            # also try metadata.name at same level
            m_meta_name = re.match(r"^  name:\s+(.+)", line)
            if m_meta_name and in_metadata:
                name = m_meta_name.group(1).strip()

        label = f"{kind}/{name}" if kind and name else f"document-{i}"
        safe_label = re.sub(r"[^A-Za-z0-9_\-./]", "_", label)
        cid = f"k8s::{slug}::{safe_label}"

        text = (
            f"Kubernetes manifest: {label}  (in {slug})\n"
            f"Kind: {kind or 'unknown'}\n"
            f"Name: {name or 'unknown'}\n\n"
            f"Source:\n{doc[:2000]}"
        )
        chunks.append(_make_chunk(cid, "k8s_manifest", text, {
            "path": slug,
            "kind": kind,
            "name": name,
            "document_index": i,
        }))
    return chunks


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def can_parse(path: Path) -> bool:
    return path.suffix in _EXTENSIONS


def parse_file(path: Path, root: Path) -> list[dict]:
    """Parse a YAML file into semantic chunks."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] yaml_parser cannot read {path}: {exc}", file=sys.stderr)
        slug = path.relative_to(root).as_posix()
        return [{
            "id": f"file::{slug}",
            "type": "file",
            "language": "yaml",
            "text": f"File: {slug}\n[read error: {exc}]",
            "metadata": {"path": slug, "parse_error": str(exc)},
        }]

    slug = _slug(path, root)

    # If the file has multiple k8s documents (--- separator), route directly
    # Note: we check before splitting top_keys so `---` at the top doesn't confuse us.
    non_comment_lines = [l for l in source.splitlines() if l.strip() and not l.strip().startswith("#")]
    # A file is a k8s multi-doc if it has at least one `---` separator inside
    # (not just at the very start)
    stripped = source.strip()
    k8s_multi = bool(re.search(r"\n---", stripped))

    lines = source.splitlines()
    top_keys = _get_top_level_keys(lines)
    yaml_type = _detect_yaml_type(lines, top_keys)

    # Override: if has --- and kind:, definitely kubernetes
    if k8s_multi or yaml_type == "kubernetes":
        k8s_chunks = _parse_k8s_documents(source, slug)
        if k8s_chunks:
            return k8s_chunks

    chunks: list[dict] = []

    if yaml_type == "docker-compose":
        # Each service becomes a chunk
        services = _extract_children_block(lines, "services", child_indent=2)
        for svc_name, svc_block in services.items():
            cid = f"compose-service::{slug}::{svc_name}"
            text = (
                f"Docker Compose service: {svc_name}  (in {slug})\n"
                f"Configuration:\n{svc_block[:2000]}"
            )
            chunks.append(_make_chunk(cid, "compose_service", text, {
                "path": slug,
                "service": svc_name,
            }))
        if not chunks:
            # fallback to generic if no services found
            yaml_type = "generic"
        else:
            # Also add a file overview
            file_text = (
                f"File: {slug}\n"
                f"Type: Docker Compose\n"
                f"Services: {', '.join(services.keys())}\n"
                f"Top-level keys: {', '.join(top_keys)}"
            )
            chunks.insert(0, _make_chunk(f"file::{slug}", "file", file_text, {
                "path": slug,
                "yaml_type": "docker-compose",
                "services": list(services.keys()),
                "top_level_keys": top_keys,
            }))
            return chunks

    if yaml_type == "github-actions":
        # Each job becomes a chunk
        jobs = _extract_children_block(lines, "jobs", child_indent=2)
        for job_name, job_block in jobs.items():
            cid = f"gh-action-job::{slug}::{job_name}"
            text = (
                f"GitHub Actions job: {job_name}  (in {slug})\n"
                f"Configuration:\n{job_block[:2000]}"
            )
            chunks.append(_make_chunk(cid, "gh_action_job", text, {
                "path": slug,
                "job": job_name,
            }))
        if not chunks:
            yaml_type = "generic"
        else:
            file_text = (
                f"File: {slug}\n"
                f"Type: GitHub Actions workflow\n"
                f"Jobs: {', '.join(jobs.keys())}\n"
                f"Top-level keys: {', '.join(top_keys)}"
            )
            chunks.insert(0, _make_chunk(f"file::{slug}", "file", file_text, {
                "path": slug,
                "yaml_type": "github-actions",
                "jobs": list(jobs.keys()),
                "top_level_keys": top_keys,
            }))
            return chunks

    # Generic YAML fallback (or if specific parsers found nothing)
    preview = source[:2000]
    text = (
        f"Config file: {slug}\n"
        f"Language: YAML\n"
        f"Top-level keys: {', '.join(top_keys) if top_keys else '(none parsed)'}\n\n"
        f"Content:\n{preview}"
    )
    return [_make_chunk(f"file::{slug}", "file", text, {
        "path": slug,
        "yaml_type": yaml_type,
        "top_level_keys": top_keys,
    })]
