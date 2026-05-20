"""
embed_chunks.py — Enrich repo_chunks.jsonl with semantic embeddings via Ollama.

Reads an existing repo_chunks.jsonl, calls Ollama's /api/embeddings for each
chunk, and writes a new repo_chunks_embedded.jsonl with an extra "embedding"
field on every line.

Run this once per repo, then use ask_repo_local.py --embed for semantic search.

Usage:
    python embed_chunks.py <chunks.jsonl>
    python embed_chunks.py <chunks.jsonl> --output embedded.jsonl
    python embed_chunks.py <chunks.jsonl> --model nomic-embed-text
    python embed_chunks.py <chunks.jsonl> --resume                    # skip already-embedded chunks
    python embed_chunks.py <chunks.jsonl> --incremental               # skip unchanged source files
    python embed_chunks.py <chunks.jsonl> --incremental --repo-root ../my-repo  # source files in another dir

Requires:
    ollama pull nomic-embed-text   (~270 MB, fast)
    Ollama running: ollama serve

Zero external dependencies — uses only urllib and hashlib from stdlib.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Ollama embedding call
# ---------------------------------------------------------------------------

def embed_text(text: str, model: str, base_url: str) -> list[float]:
    """Call Ollama /api/embeddings and return the embedding vector."""
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["embedding"]


# ---------------------------------------------------------------------------
# Incremental hashing helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file's content.  Returns 'sha256:<hex>'."""
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except Exception:
        return "sha256:UNREADABLE"
    return f"sha256:{h.hexdigest()}"


def _sidecar_path(output_path: Path) -> Path:
    """Return the path for the file-hashes sidecar JSON."""
    return output_path.with_name(output_path.stem + "_file_hashes.json")


def _load_hashes(sidecar: Path) -> dict[str, str]:
    """Load existing file hashes from sidecar, or return empty dict."""
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_hashes(sidecar: Path, hashes: dict[str, str]) -> None:
    """Write updated file hashes to the sidecar file."""
    sidecar.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")


def _source_file_for_chunk(chunk: dict) -> str | None:
    """
    Return the source file slug for a chunk, or None if not applicable.
    Used to group chunks by source file for incremental re-embedding.
    """
    meta = chunk.get("metadata", {})
    # function / class chunks have "file"
    if "file" in meta:
        return meta["file"]
    # file chunks have "path"
    if "path" in meta:
        return meta["path"]
    # repo_overview, usages, doc, config, file_js etc. — no source file tracking
    return None


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    model    = _pop_arg(args, "--model")     or "nomic-embed-text"
    base_url = _pop_arg(args, "--url")       or "http://localhost:11434"
    output   = _pop_arg(args, "--output")
    repo_root = _pop_arg(args, "--repo-root")
    resume   = "--resume" in args
    incremental = "--incremental" in args
    if resume:
        args.remove("--resume")
    if incremental:
        args.remove("--incremental")

    input_path = Path(args[0])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output) if output else Path(
        input_path.stem + "_embedded.jsonl"
    )

    # load input chunks
    chunks = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {input_path}")
    print(f"Embedding model : {model}  (Ollama at {base_url})")
    print(f"Output          : {output_path}")

    # ---- Incremental mode: detect changed files ----
    # Maps source-file slug -> whether to skip it (True = skip / reuse old embedding)
    skip_file: dict[str, bool] = {}
    # Will hold {id -> embedded_chunk} from the existing output for reuse
    existing_by_id: dict[str, dict] = {}

    if incremental:
        sidecar = _sidecar_path(output_path)
        old_hashes = _load_hashes(sidecar)
        new_hashes: dict[str, str] = {}

        # Determine the repo root: we need to resolve source file paths.
        # Check --repo-root first, then input_path's parent, then cwd.
        search_roots = []
        if repo_root:
            search_roots.append(Path(repo_root).resolve())
        search_roots += [input_path.parent, Path(".").resolve()]

        # Collect all distinct source file slugs from the chunks
        source_files: set[str] = set()
        for chunk in chunks:
            sf = _source_file_for_chunk(chunk)
            if sf:
                source_files.add(sf)

        # Compute current hashes for each source file
        files_unchanged = 0
        files_changed = 0
        files_not_found = 0

        for sf in sorted(source_files):
            resolved: Path | None = None
            for root in search_roots:
                candidate = root / sf
                if candidate.exists():
                    resolved = candidate
                    break

            if resolved is None:
                # Can't find the file — treat as changed (will re-embed)
                files_not_found += 1
                new_hashes[sf] = "sha256:NOT_FOUND"
                skip_file[sf] = False
                continue

            current_hash = _hash_file(resolved)
            new_hashes[sf] = current_hash

            if old_hashes.get(sf) == current_hash:
                skip_file[sf] = True
                files_unchanged += 1
            else:
                skip_file[sf] = False
                files_changed += 1

        # Load existing embeddings for reuse
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_by_id[obj["id"]] = obj
                except Exception:
                    pass

        print(f"\nIncremental mode:")
        print(f"  {files_unchanged} files unchanged (embeddings will be reused)")
        print(f"  {files_changed} files changed/new (will be re-embedded)")
        if files_not_found:
            print(f"  {files_not_found} source files not found on disk (treated as changed)")
        print()

    # ---- --resume mode: load already-embedded ids ----
    already_done: set[str] = set()
    existing_lines: list[str] = []
    if resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                existing_lines.append(line)
                obj = json.loads(line)
                already_done.add(obj["id"])
        print(f"Resuming: {len(already_done)} chunks already embedded, skipping them.")

    # verify Ollama is reachable
    try:
        embed_text("test", model, base_url)
    except urllib.error.URLError as exc:
        print(f"\nError: cannot reach Ollama at {base_url}: {exc}", file=sys.stderr)
        print("Make sure Ollama is running:  ollama serve", file=sys.stderr)
        print(f"And the model is available:   ollama pull {model}", file=sys.stderr)
        sys.exit(1)
    except KeyError:
        print(f"\nError: model '{model}' not found in Ollama.", file=sys.stderr)
        print(f"Run:  ollama pull {model}", file=sys.stderr)
        sys.exit(1)

    print()

    # open output file (append if resuming, overwrite otherwise)
    mode = "a" if resume else "w"
    total = len(chunks)
    done = len(already_done)
    errors = 0
    reused = 0
    t0 = time.time()

    with output_path.open(mode, encoding="utf-8") as fh:
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk["id"]

            # --resume: skip already-done
            if chunk_id in already_done:
                continue

            # --incremental: reuse existing embedding if source file unchanged
            if incremental:
                sf = _source_file_for_chunk(chunk)
                if sf is not None and skip_file.get(sf, False):
                    if chunk_id in existing_by_id:
                        fh.write(json.dumps(existing_by_id[chunk_id], ensure_ascii=False) + "\n")
                        reused += 1
                        done += 1
                        if i % 50 == 0 or i == total:
                            elapsed = time.time() - t0
                            rate = done / elapsed if elapsed > 0 else 0
                            remaining = (total - i) / rate if rate > 0 else 0
                            print(
                                f"  {i}/{total} chunks processed "
                                f"| {rate:.1f} chunks/s "
                                f"| ETA {remaining:.0f}s  [{reused} reused]"
                            )
                        continue
                    # If we don't have it in existing, fall through to embed

            try:
                embedding = embed_text(chunk["text"], model, base_url)
                chunk_out = dict(chunk)
                chunk_out["embedding"] = embedding
                fh.write(json.dumps(chunk_out, ensure_ascii=False) + "\n")
                done += 1
            except Exception as exc:
                print(f"  [warn] chunk {chunk_id}: {exc}", file=sys.stderr)
                # write chunk without embedding so we don't lose it
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                errors += 1

            # progress every 50 chunks
            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                reused_label = f"  [{reused} reused]" if incremental else ""
                print(
                    f"  {i}/{total} chunks embedded "
                    f"| {rate:.1f} chunks/s "
                    f"| ETA {remaining:.0f}s{reused_label}"
                )

    # Save updated hashes sidecar for incremental mode
    if incremental:
        _save_hashes(_sidecar_path(output_path), new_hashes)
        files_unchanged_count = sum(1 for v in skip_file.values() if v)
        files_changed_count = sum(1 for v in skip_file.values() if not v)
        print(f"\nSummary: {files_unchanged_count} files unchanged (embeddings reused), "
              f"{files_changed_count} files changed/new (re-embedded), "
              f"{total} chunks total")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Wrote {done} embedded chunks -> {output_path}")
    if errors:
        print(f"  {errors} chunks skipped due to errors (written without embedding)")
    print(f"\nNow query with:")
    print(f"  python ask_repo_local.py {output_path} \"your question\" --embed --model <llm>")


if __name__ == "__main__":
    main()
