"""
chroma_store.py — ChromaDB persistent vector store for CodeIntelligence.

Drop-in alternative to the in-memory numpy cosine similarity used by default.
Persists embeddings across restarts so you pay the embedding cost only once.

Typical usage via Python:
    from chroma_store import build_chroma_index, query_chroma
    collection = build_chroma_index("repo_chunks_embedded.jsonl")
    results = query_chroma(collection, query_embedding, top_k=6)

Standalone CLI:
    python chroma_store.py build  repo_chunks_embedded.jsonl
    python chroma_store.py query  repo_chunks_embedded.jsonl "how does auth work" --top-k 5
    python chroma_store.py info   repo_chunks_embedded.jsonl

Requirements:
    pip install chromadb

Multiple repos coexist in the same Chroma instance because the collection name
is derived from the JSONL filename (e.g. "repo_chunks_embedded" for
"repo_chunks_embedded.jsonl").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_PERSIST_DIR = ".codeintelligence_chroma"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _require_chromadb():
    """Import chromadb or raise a helpful ImportError."""
    try:
        import chromadb  # type: ignore
        return chromadb
    except ImportError:
        raise ImportError(
            "chromadb is not installed.  Install it with:\n"
            "    pip install chromadb\n"
            "Then re-run the command."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collection_name_from_path(jsonl_path: str) -> str:
    """Derive a clean Chroma collection name from the JSONL filename."""
    stem = Path(jsonl_path).stem  # e.g. "repo_chunks_embedded"
    # Chroma collection names must be 3-63 chars, alphanumeric + underscore/hyphen
    name = stem.replace(" ", "_").replace(".", "_")
    # Truncate to 63 characters if needed
    return name[:63]


def _load_jsonl(path: str) -> List[dict]:
    chunks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_collection(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = "codeintelligence",
):
    """
    Return an existing Chroma collection or create it.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB persists its data.
    collection_name:
        Name of the collection inside ChromaDB.

    Returns
    -------
    chromadb.Collection object.
    """
    chromadb = _require_chromadb()
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def build_chroma_index(
    chunks_path: str,
    collection_name: Optional[str] = None,
    persist_dir: Optional[str] = None,
):
    """
    Load an embedded JSONL and upsert all chunks into ChromaDB.

    This function is idempotent — safe to re-run on the same file.
    Chunks without an ``"embedding"`` field are skipped.

    Parameters
    ----------
    chunks_path:
        Path to the embedded JSONL file produced by embed_chunks.py.
    collection_name:
        Chroma collection name.  Defaults to the JSONL stem.
    persist_dir:
        ChromaDB persistence directory.  Defaults to ``.codeintelligence_chroma/``.

    Returns
    -------
    chromadb.Collection object ready for querying.
    """
    _persist_dir = persist_dir or DEFAULT_PERSIST_DIR
    _collection_name = collection_name or _collection_name_from_path(chunks_path)

    print(f"Loading chunks from {chunks_path} …")
    chunks = _load_jsonl(chunks_path)
    embedded = [c for c in chunks if "embedding" in c]
    skipped = len(chunks) - len(embedded)
    print(f"  {len(chunks)} chunks total, {len(embedded)} have embeddings"
          + (f", {skipped} skipped (no embedding)" if skipped else ""))

    if not embedded:
        raise ValueError(
            f"No chunks with embeddings found in {chunks_path}.\n"
            "Run embed_chunks.py first to generate embeddings."
        )

    collection = get_or_create_collection(_persist_dir, _collection_name)

    # Upsert in batches to keep memory usage reasonable
    batch_size = 500
    total_upserted = 0
    for start in range(0, len(embedded), batch_size):
        batch = embedded[start: start + batch_size]
        ids = [c["id"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        # Store everything except the heavy embedding in metadata
        metadatas = []
        documents = []
        for c in batch:
            doc = c.get("text", "")[:2000]  # Chroma has per-document limits
            documents.append(doc)
            meta = {k: v for k, v in c.items() if k not in ("embedding", "text")}
            # Chroma metadata values must be str, int, float, or bool
            safe_meta: dict = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                elif isinstance(v, dict):
                    safe_meta[k] = json.dumps(v)
                elif isinstance(v, list):
                    safe_meta[k] = json.dumps(v)
                else:
                    safe_meta[k] = str(v)
            metadatas.append(safe_meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total_upserted += len(batch)
        print(f"  Upserted {total_upserted}/{len(embedded)} …")

    print(f"ChromaDB index ready: collection={_collection_name!r} "
          f"in {_persist_dir!r}  ({collection.count()} docs total)")
    return collection


def query_chroma(
    collection,
    query_embedding: List[float],
    top_k: int = 10,
) -> List[dict]:
    """
    Return top_k chunks by cosine similarity from a Chroma collection.

    Returns chunks in the same dict format as the existing numpy retrieval
    (keys: ``id``, ``type``, ``text``, ``metadata``, optionally ``score``).

    Parameters
    ----------
    collection:
        chromadb.Collection returned by build_chroma_index or get_or_create_collection.
    query_embedding:
        Float list of the query embedding vector.
    top_k:
        Number of results to return.

    Returns
    -------
    List of chunk dicts sorted by similarity (highest first).
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[dict] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
        # Convert Chroma distance (lower = more similar for cosine) to score
        score = 1.0 - dist  # cosine distance → cosine similarity

        # Reconstruct metadata dict (unpack JSON strings if needed)
        reconstructed_meta: dict = {}
        for k, v in (meta or {}).items():
            if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                try:
                    reconstructed_meta[k] = json.loads(v)
                    continue
                except Exception:
                    pass
            reconstructed_meta[k] = v

        chunk = {
            "id": chunk_id,
            "type": reconstructed_meta.pop("type", "unknown"),
            "text": doc,
            "metadata": reconstructed_meta,
            "score": score,
        }
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _cli_build(jsonl_path: str, persist_dir: str):
    build_chroma_index(jsonl_path, persist_dir=persist_dir)


def _cli_query(jsonl_path: str, question: str, top_k: int, persist_dir: str):
    """Query the index using an embedded question via Ollama."""
    import urllib.request

    collection_name = _collection_name_from_path(jsonl_path)
    collection = get_or_create_collection(persist_dir, collection_name)

    if collection.count() == 0:
        print("Collection is empty — run 'build' first.", file=sys.stderr)
        sys.exit(1)

    # Embed the query via Ollama
    ollama_url = "http://localhost:11434"
    embed_model = "nomic-embed-text"
    payload = json.dumps({"model": embed_model, "prompt": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_url}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        query_vec = result["embedding"]
    except Exception as exc:
        print(f"Failed to embed query via Ollama: {exc}", file=sys.stderr)
        sys.exit(1)

    results = query_chroma(collection, query_vec, top_k=top_k)
    print(f"\nTop {len(results)} results for: {question!r}\n")
    for i, chunk in enumerate(results, 1):
        score = chunk.get("score", 0.0)
        print(f"  [{i}] score={score:.4f}  id={chunk['id']}")
        preview = chunk["text"][:200].replace("\n", " ")
        print(f"       {preview}…\n")


def _cli_info(jsonl_path: str, persist_dir: str):
    collection_name = _collection_name_from_path(jsonl_path)
    print(f"Collection name : {collection_name}")
    print(f"Persist dir     : {persist_dir}")
    try:
        collection = get_or_create_collection(persist_dir, collection_name)
        print(f"Documents       : {collection.count()}")
    except Exception as exc:
        print(f"Could not open collection: {exc}")


def _pop_arg(args: List[str], flag: str, default=None):
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1] if idx + 1 < len(args) else default
        del args[idx: idx + 2]
        return val
    return default


def main():
    args = sys.argv[1:]

    if len(args) < 2 or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Usage:")
        print("  python chroma_store.py build <embedded.jsonl> [--persist-dir DIR]")
        print("  python chroma_store.py query <embedded.jsonl> <question> [--top-k N] [--persist-dir DIR]")
        print("  python chroma_store.py info  <embedded.jsonl> [--persist-dir DIR]")
        sys.exit(0)

    persist_dir = _pop_arg(args, "--persist-dir") or DEFAULT_PERSIST_DIR
    top_k = int(_pop_arg(args, "--top-k") or 5)

    command = args[0]
    jsonl_path = args[1] if len(args) > 1 else None

    if not jsonl_path:
        print("Error: missing <embedded.jsonl> argument", file=sys.stderr)
        sys.exit(1)

    if command == "build":
        _cli_build(jsonl_path, persist_dir)

    elif command == "query":
        if len(args) < 3:
            print("Error: missing question", file=sys.stderr)
            sys.exit(1)
        question = args[2]
        _cli_query(jsonl_path, question, top_k, persist_dir)

    elif command == "info":
        _cli_info(jsonl_path, persist_dir)

    else:
        print(f"Unknown command: {command!r}. Use build, query, or info.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
