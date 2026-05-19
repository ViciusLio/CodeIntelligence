"""
ask_repo.py — Query a Python repository using RAG + Claude.

Loads repo_chunks.jsonl produced by parse_repo.py, retrieves the most
relevant chunks for your question using TF-IDF, and answers via Claude API.

Usage:
    python ask_repo.py <chunks.jsonl> "<question>"
    python ask_repo.py <chunks.jsonl>              # interactive mode
    python ask_repo.py repo_chunks.jsonl "Where is JWT token creation implemented?"

Options:
    --api-key <key>    Anthropic API key (or set ANTHROPIC_API_KEY env var)
    --top-k <n>        Number of chunks to retrieve per query (default: 12)
    --all-chunks       Send all chunks as context (broad / overview questions)
    --model <id>       Claude model to use (default: claude-opus-4-7)
    --no-stream        Disable streaming output

Requires:
    pip install anthropic
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def load_chunks(jsonl_path: str) -> list[dict]:
    """Load all chunks from a JSONL file."""
    chunks: list[dict] = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF retrieval (stdlib only)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _tf_score(query_tokens: list[str], text: str) -> float:
    """
    Length-normalised TF score.
    Raw hit count divided by document length (in tokens) so that large chunks
    don't dominate just by being longer than smaller, more focused ones.
    Multiplied by 1000 to keep numbers readable.
    """
    text_lower = text.lower()
    raw = sum(text_lower.count(tok) for tok in query_tokens)
    doc_len = max(len(re.findall(r"\w+", text_lower)), 1)
    return raw / doc_len * 1000


def retrieve(chunks: list[dict], question: str, top_k: int = 12) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a question using TF-IDF.
    The repo_overview chunk is always included (prepended if not already in top-k).
    """
    query_tokens = _tokenize(question)
    scored = [(c, _tf_score(query_tokens, c["text"])) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = [c for c, _ in scored[:top_k]]

    # always include repo_overview for grounding
    overview = next((c for c in chunks if c["type"] == "repo_overview"), None)
    if overview and overview not in top:
        top = [overview] + top[: top_k - 1]

    return top


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a prompt-ready context block."""
    parts: list[str] = []
    for c in chunks:
        header = f"[{c['type'].upper()}] {c['id']}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Python software engineer and code analyst.
Answer questions about the Python repository using ONLY the context chunks provided.
Each chunk is labelled with its type (REPO_OVERVIEW, FILE, FUNCTION, CLASS) and its id.

Guidelines:
- Be precise and cite the relevant file/function/class when possible.
- If the answer requires tracing code across multiple files or layers, do so step by step.
- If the answer is not present in the provided context, say so explicitly.
- Keep answers concise but complete — include relevant code snippets when helpful.
"""


def ask(
    question: str,
    chunks: list[dict],
    api_key: str,
    top_k: int = 12,
    all_chunks: bool = False,
    model: str = "claude-opus-4-7",
    stream: bool = True,
) -> tuple[str, int]:
    """
    Send a question to Claude with retrieved context.
    Returns (answer_text, number_of_chunks_used).
    """
    import anthropic

    relevant = chunks if all_chunks else retrieve(chunks, question, top_k)
    context = build_context(relevant)
    n_chunks = len(relevant)

    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"Context from the Python repository:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )

    if stream:
        # streaming: print tokens as they arrive, collect full text
        full_text = ""
        with client.messages.stream(
            model=model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream_ctx:
            for text in stream_ctx.text_stream:
                print(text, end="", flush=True)
                full_text += text
        print()  # newline after streamed output
        return full_text, n_chunks
    else:
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        # extract text from content blocks (skip thinking blocks)
        text_parts = [
            b.text for b in message.content if hasattr(b, "text")
        ]
        return "".join(text_parts), n_chunks


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

    api_key = _pop_arg(args, "--api-key") or os.environ.get("ANTHROPIC_API_KEY", "")
    top_k = int(_pop_arg(args, "--top-k") or 12)
    model = _pop_arg(args, "--model") or "claude-opus-4-7"
    all_chunks = "--all-chunks" in args
    no_stream = "--no-stream" in args
    if all_chunks:
        args.remove("--all-chunks")
    if no_stream:
        args.remove("--no-stream")

    if not api_key:
        print("Error: set ANTHROPIC_API_KEY or pass --api-key <key>", file=sys.stderr)
        sys.exit(1)

    if not args:
        print("Error: missing <chunks.jsonl> argument", file=sys.stderr)
        print(__doc__)
        sys.exit(1)

    jsonl_path = args[0]
    question = args[1] if len(args) > 1 else None

    if not Path(jsonl_path).exists():
        print(f"Error: file not found: {jsonl_path!r}", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(jsonl_path)
    print(f"Loaded {len(chunks)} chunks from {jsonl_path}")
    print(f"Model: {model}  |  top_k: {top_k}  |  stream: {not no_stream}")
    print()

    if question:
        # single-shot mode
        print(f"Q: {question}")
        print()
        answer, n = ask(
            question, chunks, api_key,
            top_k=top_k, all_chunks=all_chunks,
            model=model, stream=not no_stream,
        )
        if no_stream:
            # in non-stream mode the answer wasn't printed yet
            print(f"A: {answer}")
        print(f"\n[{n} chunks used]")
    else:
        # interactive mode
        print("Interactive mode — type your question (Ctrl+C or empty line × 2 to exit)\n")
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
            answer, n = ask(
                q, chunks, api_key,
                top_k=top_k, all_chunks=all_chunks,
                model=model, stream=not no_stream,
            )
            if no_stream:
                print(f"A: {answer}")
            print(f"\n[{n} chunks used]\n")


if __name__ == "__main__":
    main()
