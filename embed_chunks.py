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
    python embed_chunks.py <chunks.jsonl> --resume   # skip already-embedded chunks

Requires:
    ollama pull nomic-embed-text   (~270 MB, fast)
    Ollama running: ollama serve

Zero external dependencies — uses only urllib from stdlib.
"""

from __future__ import annotations

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

    model    = _pop_arg(args, "--model")  or "nomic-embed-text"
    base_url = _pop_arg(args, "--url")    or "http://localhost:11434"
    output   = _pop_arg(args, "--output")
    resume   = "--resume" in args
    if resume:
        args.remove("--resume")

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

    # if resuming, load already-embedded ids
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
    t0 = time.time()

    with output_path.open(mode, encoding="utf-8") as fh:
        for i, chunk in enumerate(chunks, 1):
            if chunk["id"] in already_done:
                continue

            try:
                embedding = embed_text(chunk["text"], model, base_url)
                chunk_out = dict(chunk)
                chunk_out["embedding"] = embedding
                fh.write(json.dumps(chunk_out, ensure_ascii=False) + "\n")
                done += 1
            except Exception as exc:
                print(f"  [warn] chunk {chunk['id']}: {exc}", file=sys.stderr)
                # write chunk without embedding so we don't lose it
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                errors += 1

            # progress every 50 chunks
            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{total} chunks embedded "
                    f"| {rate:.1f} chunks/s "
                    f"| ETA {remaining:.0f}s"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Wrote {done} embedded chunks -> {output_path}")
    if errors:
        print(f"  {errors} chunks skipped due to errors (written without embedding)")
    print(f"\nNow query with:")
    print(f"  python ask_repo_local.py {output_path} \"your question\" --embed --model <llm>")


if __name__ == "__main__":
    main()
