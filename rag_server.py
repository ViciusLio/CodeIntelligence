"""
rag_server.py — RAG server with OpenAI-compatible AND Ollama-compatible API.

Loads repo_chunks.jsonl, runs TF-IDF or semantic retrieval, proxies generation
to a local Ollama instance, and exposes the result on two standard API formats
so any frontend (Open WebUI, LibreChat, Chatbox, custom UI...) can connect.

Usage:
    python rag_server.py <chunks.jsonl> [options]

Options:
    --port     <n>     Listening port          (default: 8080)
    --model    <name>  Ollama LLM model        (default: qwen2.5-coder:7b)
    --ollama   <url>   Ollama base URL         (default: http://localhost:11434)
    --top-k    <n>     Chunks per query        (default: 6)
    --embed            Use semantic retrieval  (needs embed_chunks.py output)
    --embed-model <m>  Embedding model         (default: nomic-embed-text)

Endpoints exposed (both formats simultaneously):

  OpenAI-compatible  →  any OpenAI SDK / frontend just sets base_url
    GET  /v1/models
    POST /v1/chat/completions        (streaming + non-streaming)

  Ollama-compatible  →  any Ollama frontend just points to this server
    GET  /api/tags
    POST /api/chat                   (streaming + non-streaming)
    POST /api/generate               (streaming + non-streaming)

  Utility
    GET  /health
    POST /query                      (simple JSON: {"question":"...", "top_k":6})

Examples:

  # Start the server
  python rag_server.py ci_bench_L2_chunks.jsonl --port 8080 --model qwen2.5-coder:7b

  # With semantic retrieval
  python rag_server.py ci_bench_L2_chunks_embedded.jsonl --port 8080 --embed

  # Test with curl (OpenAI format)
  curl http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"rag","messages":[{"role":"user","content":"Where is JWT implemented?"}],"stream":false}'

  # Test with curl (Ollama format)
  curl http://localhost:8080/api/chat \\
    -H "Content-Type: application/json" \\
    -d '{"model":"rag","messages":[{"role":"user","content":"Where is JWT implemented?"}],"stream":false}'

  # Connect Open WebUI: set Ollama URL to http://localhost:8080
  # Connect any OpenAI client: set base_url to http://localhost:8080/v1, api_key to "rag"

Zero external dependencies — stdlib only (http.server, urllib, json, threading).
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


# ---------------------------------------------------------------------------
# Global server config (set at startup)
# ---------------------------------------------------------------------------

CONFIG: dict = {}
CHUNKS: list[dict] = []


# ---------------------------------------------------------------------------
# Retrieval (identical to ask_repo_local.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _tf_score(query_tokens: list[str], text: str) -> float:
    text_lower = text.lower()
    raw = sum(text_lower.count(tok) for tok in query_tokens)
    doc_len = max(len(re.findall(r"\w+", text_lower)), 1)
    return raw / doc_len * 1000


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def _embed_text(text: str) -> list[float]:
    payload = json.dumps({
        "model": CONFIG["embed_model"],
        "prompt": text,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{CONFIG['ollama_url']}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["embedding"]


def retrieve(question: str) -> list[dict]:
    top_k = CONFIG["top_k"]
    chunks = CHUNKS

    if CONFIG["use_embed"]:
        try:
            qvec = _embed_text(question)
            scored = [
                (c, _cosine(qvec, c["embedding"]))
                for c in chunks if "embedding" in c
            ]
        except Exception:
            scored = [(c, _tf_score(_tokenize(question), c["text"])) for c in chunks]
    else:
        tokens = _tokenize(question)
        scored = [(c, _tf_score(tokens, c["text"])) for c in chunks]

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in scored[:top_k]]
    overview = next((c for c in chunks if c["type"] == "repo_overview"), None)
    if overview and overview not in top:
        top = [overview] + top[: top_k - 1]
    return top


def build_context(chunks: list[dict]) -> str:
    parts = [f"[{c['type'].upper()}] {c['id']}\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ollama generation proxy
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python software engineer and code analyst. "
    "Answer questions about the Python repository using ONLY the context chunks provided. "
    "Each chunk is labelled with its type (REPO_OVERVIEW, FILE, FUNCTION, CLASS) and id. "
    "Be precise, cite file/function/class names when relevant. "
    "If the answer is not in the context, say so explicitly."
)


def _ollama_chat_stream(messages: list[dict]):
    """Generator: yields raw bytes lines from Ollama streaming response."""
    payload = json.dumps({
        "model": CONFIG["model"],
        "stream": True,
        "messages": messages,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{CONFIG['ollama_url']}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            yield line


def _ollama_chat_complete(messages: list[dict]) -> str:
    """Non-streaming: return full answer string."""
    payload = json.dumps({
        "model": CONFIG["model"],
        "stream": False,
        "messages": messages,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{CONFIG['ollama_url']}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        obj = json.loads(resp.read())
    return obj.get("message", {}).get("content", "")


def build_messages(question: str, history: list[dict] | None = None) -> list[dict]:
    """Build Ollama messages list with RAG context injected."""
    relevant = retrieve(question)
    context = build_context(relevant)
    augmented_question = (
        f"Context from the Python repository:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # include prior turns (without re-injecting context)
        for m in history[:-1]:
            msgs.append(m)
    msgs.append({"role": "user", "content": augmented_question})
    return msgs, relevant


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class RAGHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    # ---- routing ----

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "chunks": len(CHUNKS), "model": CONFIG["model"]})
        elif self.path == "/v1/models":
            self._openai_models()
        elif self.path == "/api/tags":
            self._ollama_tags()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_body()
        if self.path == "/v1/chat/completions":
            self._openai_chat(body)
        elif self.path == "/api/chat":
            self._ollama_chat(body)
        elif self.path == "/api/generate":
            self._ollama_generate(body)
        elif self.path == "/query":
            self._simple_query(body)
        else:
            self._json({"error": "not found"}, 404)

    # ---- helpers ----

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _json(self, obj: dict, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _write(self, data: str):
        self.wfile.write(data.encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    # ---- OpenAI endpoints ----

    def _openai_models(self):
        self._json({
            "object": "list",
            "data": [{
                "id": "rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }]
        })

    def _openai_chat(self, body: dict):
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if not question:
            self._json({"error": "no user message"}, 400)
            return

        ollama_msgs, relevant = build_messages(question, messages)
        comp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        if stream:
            self._sse_start()
            # opening delta with role
            chunk = {
                "id": comp_id, "object": "chat.completion.chunk",
                "created": created, "model": "rag",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            self._write(f"data: {json.dumps(chunk)}\n\n")
            # stream tokens
            for raw_line in _ollama_chat_stream(ollama_msgs):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                done = obj.get("done", False)
                if token:
                    chunk = {
                        "id": comp_id, "object": "chat.completion.chunk",
                        "created": created, "model": "rag",
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                    self._write(f"data: {json.dumps(chunk)}\n\n")
                if done:
                    break
            # closing chunk
            chunk = {
                "id": comp_id, "object": "chat.completion.chunk",
                "created": created, "model": "rag",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            self._write(f"data: {json.dumps(chunk)}\n\n")
            self._write("data: [DONE]\n\n")
        else:
            answer = _ollama_chat_complete(ollama_msgs)
            self._json({
                "id": comp_id,
                "object": "chat.completion",
                "created": created,
                "model": "rag",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
                "_rag_chunks_used": len(relevant),
            })

    # ---- Ollama endpoints ----

    def _ollama_tags(self):
        self._json({
            "models": [{
                "name": "rag",
                "model": "rag",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 0,
                "details": {"family": "rag", "parameter_size": "RAG"},
            }]
        })

    def _ollama_chat(self, body: dict):
        messages = body.get("messages", [])
        stream = body.get("stream", True)
        question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if not question:
            self._json({"error": "no user message"}, 400)
            return

        ollama_msgs, relevant = build_messages(question, messages)
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if stream:
            self._sse_start()
            for raw_line in _ollama_chat_stream(ollama_msgs):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                done = obj.get("done", False)
                out = {
                    "model": "rag",
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": token},
                    "done": done,
                }
                self._write(json.dumps(out) + "\n")
                if done:
                    break
        else:
            answer = _ollama_chat_complete(ollama_msgs)
            self._json({
                "model": "rag",
                "created_at": created_at,
                "message": {"role": "assistant", "content": answer},
                "done": True,
                "_rag_chunks_used": len(relevant),
            })

    def _ollama_generate(self, body: dict):
        """Single-turn /api/generate (no message history)."""
        prompt = body.get("prompt", "")
        stream = body.get("stream", True)
        if not prompt:
            self._json({"error": "missing prompt"}, 400)
            return
        # wrap as chat
        body["messages"] = [{"role": "user", "content": prompt}]
        self._ollama_chat(body)

    # ---- Simple /query endpoint ----

    def _simple_query(self, body: dict):
        question = body.get("question", "")
        top_k_override = body.get("top_k")
        if top_k_override:
            CONFIG["top_k"] = int(top_k_override)
        if not question:
            self._json({"error": "missing question"}, 400)
            return
        ollama_msgs, relevant = build_messages(question)
        answer = _ollama_chat_complete(ollama_msgs)
        self._json({
            "answer": answer,
            "sources": [
                {"id": c["id"], "type": c["type"],
                 "content": c["text"][:300] + ("..." if len(c["text"]) > 300 else "")}
                for c in relevant
            ],
        })


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

    port        = int(_pop_arg(args, "--port")        or 8080)
    model       = _pop_arg(args, "--model")           or "qwen2.5-coder:7b"
    ollama_url  = _pop_arg(args, "--ollama")          or "http://localhost:11434"
    top_k       = int(_pop_arg(args, "--top-k")       or 6)
    embed_model = _pop_arg(args, "--embed-model")     or "nomic-embed-text"
    use_embed   = "--embed" in args
    if use_embed:
        args.remove("--embed")

    if not args:
        print("Error: missing <chunks.jsonl>", file=sys.stderr)
        sys.exit(1)

    jsonl_path = Path(args[0])
    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    # load chunks into global
    global CHUNKS, CONFIG
    CHUNKS = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            CHUNKS.append(json.loads(line))

    CONFIG = {
        "model": model,
        "ollama_url": ollama_url,
        "top_k": top_k,
        "use_embed": use_embed,
        "embed_model": embed_model,
    }

    retrieval = f"semantic ({embed_model})" if use_embed else "TF-IDF"

    print(f"RAG Server starting...")
    print(f"  Chunks     : {len(CHUNKS)} from {jsonl_path.name}")
    print(f"  LLM model  : {model}  (via Ollama at {ollama_url})")
    print(f"  Retrieval  : {retrieval}  |  top_k: {top_k}")
    print(f"  Port       : {port}")
    print()
    print(f"  OpenAI endpoint : http://localhost:{port}/v1/chat/completions")
    print(f"  Ollama endpoint : http://localhost:{port}/api/chat")
    print(f"  Health check    : http://localhost:{port}/health")
    print()
    print(f"  Connect Open WebUI  -> set Ollama URL to http://localhost:{port}")
    print(f"  Connect any OpenAI client -> base_url=http://localhost:{port}/v1  api_key=rag")
    print()
    print("Press Ctrl+C to stop.\n")

    server = HTTPServer(("0.0.0.0", port), RAGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
