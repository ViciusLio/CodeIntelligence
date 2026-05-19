"""
ask_repo_local.py — Query a Python repository using RAG + a local LLM (Ollama).

Supports two retrieval modes:
  - TF-IDF (default)  : keyword-based, zero setup, works on any chunks.jsonl
  - Semantic (--embed): cosine similarity on embeddings, much better quality,
                        requires running embed_chunks.py first

Usage:
    python ask_repo_local.py <chunks.jsonl> "<question>"
    python ask_repo_local.py <chunks.jsonl>              # interactive mode

Options:
    --model        <name>   Ollama LLM to use          (default: codellama)
    --embed-model  <name>   Ollama embedding model     (default: nomic-embed-text)
    --top-k        <n>      Chunks to retrieve          (default: 12)
    --url          <url>    Ollama base URL             (default: http://localhost:11434)
    --embed                 Use semantic retrieval (needs embed_chunks.py output)
    --all-chunks            Send all chunks (broad questions)
    --no-stream             Disable streaming output

Workflow with semantic retrieval:
    # Step 1 — embed once (a few minutes)
    python embed_chunks.py repo_chunks.jsonl
    # Step 2 — query with semantic search
    python ask_repo_local.py repo_chunks_embedded.jsonl "..." --embed

Popular LLM models for Ollama:
    qwen2.5-coder:7b   -- best for Python code (recommended)
    codellama          -- Meta code model
    deepseek-coder     -- strong on code
    llama3             -- general purpose
    mistral            -- fast and capable

Make sure Ollama is running:
    ollama serve
    ollama pull qwen2.5-coder:7b
    ollama pull nomic-embed-text   # only needed for --embed mode
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Chunk loading (identical to ask_repo.py)
# ---------------------------------------------------------------------------

def load_chunks(jsonl_path: str) -> list[dict]:
    chunks: list[dict] = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF retrieval (identical to ask_repo.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _tf_score(query_tokens: list[str], text: str) -> float:
    text_lower = text.lower()
    raw = sum(text_lower.count(tok) for tok in query_tokens)
    doc_len = max(len(re.findall(r"\w+", text_lower)), 1)
    return raw / doc_len * 1000


def retrieve(chunks: list[dict], question: str, top_k: int = 12) -> list[dict]:
    """TF-IDF retrieval (default, no setup needed)."""
    query_tokens = _tokenize(question)
    scored = [(c, _tf_score(query_tokens, c["text"])) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in scored[:top_k]]
    overview = next((c for c in chunks if c["type"] == "repo_overview"), None)
    if overview and overview not in top:
        top = [overview] + top[: top_k - 1]
    return top


# ---------------------------------------------------------------------------
# Semantic retrieval (cosine similarity on embeddings)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_query(question: str, model: str, base_url: str) -> list[float]:
    payload = json.dumps({"model": model, "prompt": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["embedding"]


def retrieve_semantic(
    chunks: list[dict],
    question: str,
    top_k: int = 12,
    embed_model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> list[dict]:
    """Semantic retrieval via cosine similarity on Ollama embeddings."""
    # embed the question
    try:
        query_vec = _embed_query(question, embed_model, base_url)
    except Exception as exc:
        print(f"[warn] embedding failed ({exc}), falling back to TF-IDF", file=sys.stderr)
        return retrieve(chunks, question, top_k)

    # score only chunks that have an embedding field
    scored = []
    no_embedding = []
    for c in chunks:
        if "embedding" in c:
            score = _cosine(query_vec, c["embedding"])
            scored.append((c, score))
        else:
            no_embedding.append(c)

    if not scored:
        print("[warn] no embedded chunks found, falling back to TF-IDF", file=sys.stderr)
        return retrieve(chunks, question, top_k)

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in scored[:top_k]]

    # always include repo_overview for grounding
    overview = next((c for c in chunks if c["type"] == "repo_overview"), None)
    if overview and overview not in top:
        top = [overview] + top[: top_k - 1]

    return top


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for c in chunks:
        header = f"[{c['type'].upper()}] {c['id']}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Python software engineer and code analyst.
Answer questions about the Python repository using ONLY the context chunks provided.
Each chunk is labelled with its type (REPO_OVERVIEW, FILE, FUNCTION, CLASS) and its id.

Guidelines:
- Be precise and cite the relevant file/function/class when possible.
- If the answer requires tracing code across multiple files, do so step by step.
- If the answer is not in the provided context, say so explicitly.
- Keep answers concise but complete. Include relevant code snippets when helpful.
"""


def ask_ollama(
    question: str,
    chunks: list[dict],
    model: str = "codellama",
    base_url: str = "http://localhost:11434",
    top_k: int = 12,
    all_chunks: bool = False,
    stream: bool = True,
    use_embed: bool = False,
    embed_model: str = "nomic-embed-text",
) -> tuple[str, int]:
    """
    Call Ollama's /api/chat endpoint with the retrieved context.
    Returns (answer_text, number_of_chunks_used).
    """
    if all_chunks:
        relevant = chunks
    elif use_embed:
        relevant = retrieve_semantic(chunks, question, top_k, embed_model, base_url)
    else:
        relevant = retrieve(chunks, question, top_k)
    context = build_context(relevant)
    n_chunks = len(relevant)

    user_content = (
        f"Context from the Python repository:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )

    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    url = f"{base_url.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full_text = ""
    try:
        with urllib.request.urlopen(req) as resp:
            if stream:
                # Ollama streams one JSON object per line
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_text += token
                    if obj.get("done"):
                        break
                print()  # newline after stream
            else:
                body = resp.read().decode("utf-8")
                obj = json.loads(body)
                full_text = obj.get("message", {}).get("content", "")

    except urllib.error.URLError as exc:
        print(f"\n[Error] Cannot reach Ollama at {url}: {exc}", file=sys.stderr)
        print("Make sure Ollama is running:  ollama serve", file=sys.stderr)
        sys.exit(1)

    return full_text, n_chunks


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

    model       = _pop_arg(args, "--model")       or "codellama"
    embed_model = _pop_arg(args, "--embed-model") or "nomic-embed-text"
    top_k       = int(_pop_arg(args, "--top-k")   or 12)
    base_url    = _pop_arg(args, "--url")         or "http://localhost:11434"
    all_chunks  = "--all-chunks" in args
    no_stream   = "--no-stream"  in args
    use_embed   = "--embed"      in args
    if all_chunks: args.remove("--all-chunks")
    if no_stream:  args.remove("--no-stream")
    if use_embed:  args.remove("--embed")

    if not args:
        print("Error: missing <chunks.jsonl> argument", file=sys.stderr)
        sys.exit(1)

    jsonl_path = args[0]
    question   = args[1] if len(args) > 1 else None

    if not Path(jsonl_path).exists():
        print(f"Error: file not found: {jsonl_path!r}", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(jsonl_path)
    print(f"Loaded {len(chunks)} chunks from {jsonl_path}")
    print(f"Model     : {model}  (Ollama at {base_url})")
    retrieval_mode = f"semantic ({embed_model})" if use_embed else "TF-IDF"
    print(f"Retrieval : {retrieval_mode}  |  top_k: {top_k}  |  stream: {not no_stream}")
    print()

    if question:
        print(f"Q: {question}")
        print()
        answer, n = ask_ollama(
            question, chunks,
            model=model, base_url=base_url,
            top_k=top_k, all_chunks=all_chunks,
            stream=not no_stream,
            use_embed=use_embed, embed_model=embed_model,
        )
        if no_stream:
            print(f"A: {answer}")
        print(f"\n[{n} chunks used]")
    else:
        print("Interactive mode — type your question (Ctrl+C to exit)\n")
        empty_count = 0
        while True:
            try:
                q = input("Q: ").strip()
            except KeyboardInterrupt:
                print("\nBye!")
                break
            if not q:
                empty_count += 1
                if empty_count >= 2:
                    print("Bye!")
                    break
                continue
            empty_count = 0
            print()
            answer, n = ask_ollama(
                q, chunks,
                model=model, base_url=base_url,
                top_k=top_k, all_chunks=all_chunks,
                stream=not no_stream,
                use_embed=use_embed, embed_model=embed_model,
            )
            if no_stream:
                print(f"A: {answer}")
            print(f"\n[{n} chunks used]\n")


if __name__ == "__main__":
    main()
