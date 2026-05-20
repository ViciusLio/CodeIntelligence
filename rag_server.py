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
    --rerank           Rerank with cross-encoder (requires sentence-transformers)
    --chroma           Use ChromaDB vector store (requires chromadb, implies --embed)
    --escalation-threshold <f>  Confidence score below which escalation is suggested
                                (default: 0.35)
    --claude-api-key <key>      Anthropic API key for escalation endpoint.
                                Can also be set via ANTHROPIC_API_KEY env var.
                                Without this, /query/escalate returns 503.

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
    POST /query/escalate             (confidence-based escalation to Claude API)

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
# Stores the chunks retrieved for the most recent query so the escalation
# endpoint can reference them when chunks_used_ids is empty.
LAST_CHUNKS: list[dict] = []


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
    use_rerank = CONFIG.get("use_rerank", False)

    # When reranking we over-fetch (top_k * 3) so the cross-encoder has more
    # candidates to re-score before we trim to top_k.
    candidate_k = top_k * 3 if use_rerank else top_k

    if CONFIG.get("use_chroma") and CONFIG.get("chroma_collection") is not None:
        # ChromaDB semantic retrieval
        try:
            from chroma_store import query_chroma  # type: ignore
            qvec = _embed_text(question)
            top = query_chroma(CONFIG["chroma_collection"], qvec, top_k=candidate_k)
        except Exception:
            # Fallback to numpy cosine if Chroma fails
            top = _numpy_retrieve(question, candidate_k)
    elif CONFIG["use_embed"]:
        top = _numpy_retrieve(question, candidate_k)
    else:
        tokens = _tokenize(question)
        scored = [(c, _tf_score(tokens, c["text"])) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [c for c, _ in scored[:candidate_k]]

    # Always include repo_overview for grounding
    overview = next((c for c in chunks if c["type"] == "repo_overview"), None)
    if overview and overview not in top:
        top = [overview] + top[: candidate_k - 1]

    # Cross-encoder reranking
    if use_rerank:
        try:
            from rerank import rerank  # type: ignore
            top = rerank(question, top, top_n=top_k)
        except ImportError:
            pass  # degrade gracefully

    # Trim to top_k if we didn't rerank (rerank already trims)
    if not use_rerank:
        top = top[:top_k]

    return top


def _numpy_retrieve(question: str, candidate_k: int) -> list[dict]:
    """In-memory cosine similarity retrieval (numpy-free pure Python)."""
    chunks = CHUNKS
    try:
        qvec = _embed_text(question)
        scored = [
            (c, _cosine(qvec, c["embedding"]))
            for c in chunks if "embedding" in c
        ]
    except Exception:
        scored = [(c, _tf_score(_tokenize(question), c["text"])) for c in chunks]

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:candidate_k]]


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


# ---------------------------------------------------------------------------
# Claude API generation
# ---------------------------------------------------------------------------

def _claude_chat_stream(messages: list[dict]):
    """Generator: yields text tokens from Claude streaming response."""
    import anthropic
    client = anthropic.Anthropic(api_key=CONFIG["claude_api_key"])
    # extract system from messages
    system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
    user_msgs = [m for m in messages if m["role"] != "system"]
    with client.messages.stream(
        model=CONFIG["model"],
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=system,
        messages=user_msgs,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _claude_chat_complete(messages: list[dict]) -> str:
    """Non-streaming Claude call."""
    import anthropic
    client = anthropic.Anthropic(api_key=CONFIG["claude_api_key"])
    system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
    user_msgs = [m for m in messages if m["role"] != "system"]
    response = client.messages.create(
        model=CONFIG["model"],
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=system,
        messages=user_msgs,
    )
    return "".join(b.text for b in response.content if hasattr(b, "text"))


def build_messages(question: str, history: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Build messages list with RAG context injected (works for both Ollama and Claude)."""
    global LAST_CHUNKS
    relevant = retrieve(question)
    LAST_CHUNKS = relevant   # remember for /query/escalate
    context = build_context(relevant)
    augmented_question = (
        f"Context from the Python repository:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for m in history[:-1]:
            msgs.append(m)
    msgs.append({"role": "user", "content": augmented_question})
    return msgs, relevant


def _chat_stream(messages: list[dict]):
    """Route streaming to Claude or Ollama based on config."""
    if CONFIG.get("use_claude"):
        for token in _claude_chat_stream(messages):
            yield token   # yields str tokens
    else:
        for line in _ollama_chat_stream(messages):
            yield line    # yields bytes lines


def _chat_complete(messages: list[dict]) -> str:
    """Route non-streaming to Claude or Ollama based on config."""
    if CONFIG.get("use_claude"):
        return _claude_chat_complete(messages)
    return _ollama_chat_complete(messages)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class RAGHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    # ---- routing ----

    def do_GET(self):
        if self.path in ("/", "/ui"):
            self._serve_ui()
        elif self.path == "/health":
            backend = "claude" if CONFIG.get("use_claude") else "ollama"
            reranker = "cross-encoder" if CONFIG.get("use_rerank") else "none"
            vector_store = "chroma" if CONFIG.get("use_chroma") else "numpy_in_memory"
            self._json({
                "status": "ok",
                "chunks": len(CHUNKS),
                "model": CONFIG["model"],
                "backend": backend,
                "retrieval": "semantic" if CONFIG.get("use_embed") else "tfidf",
                "reranker": reranker,
                "vector_store": vector_store,
            })
        elif self.path == "/v1/models":
            self._openai_models()
        elif self.path == "/api/tags":
            self._ollama_tags()
        else:
            self._json({"error": "not found"}, 404)

    def _serve_ui(self):
        backend = "Claude API" if CONFIG.get("use_claude") else f"Ollama · {CONFIG['model']}"
        retrieval = "Semantic" if CONFIG.get("use_embed") else "TF-IDF"
        chunks_count = len(CHUNKS)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodeIntelligence</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117; color: #e2e8f0; height: 100vh;
    display: flex; flex-direction: column;
  }}
  header {{
    padding: 14px 24px; border-bottom: 1px solid #1e2533;
    display: flex; align-items: center; justify-content: space-between;
    background: #141820;
  }}
  header h1 {{ font-size: 1.1rem; font-weight: 600; color: #7c9ef8; letter-spacing: .02em; }}
  .header-right {{ display: flex; align-items: center; gap: 12px; }}
  .badges {{ display: flex; gap: 8px; }}
  .badge {{
    font-size: 0.72rem; padding: 3px 10px; border-radius: 999px;
    background: #1e2533; color: #94a3b8;
  }}
  .badge.green {{ background: #14291f; color: #4ade80; }}
  #export-btn {{
    font-size: 0.75rem; padding: 5px 14px; border-radius: 999px;
    background: #1e2533; color: #7c9ef8; border: 1px solid #2d3748;
    cursor: pointer; transition: background .15s; white-space: nowrap;
  }}
  #export-btn:hover {{ background: #2d3748; }}
  #export-btn:disabled {{ opacity: .4; cursor: not-allowed; }}
  #chat {{
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
  }}
  .msg {{ display: flex; gap: 12px; max-width: 820px; }}
  .msg.user {{ align-self: flex-end; flex-direction: row-reverse; }}
  .avatar {{
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
  }}
  .msg.user .avatar {{ background: #3b4fd8; color: #fff; }}
  .msg.assistant .avatar {{ background: #1e2533; color: #7c9ef8; }}
  .bubble {{
    padding: 12px 16px; border-radius: 14px; line-height: 1.6;
    font-size: 0.92rem; white-space: pre-wrap; word-break: break-word;
  }}
  .msg.user .bubble {{ background: #2a3580; color: #e2e8f0; border-top-right-radius: 4px; }}
  .msg.assistant .bubble {{ background: #1a1f2e; color: #cbd5e1; border-top-left-radius: 4px; }}
  .meta {{ font-size: 0.7rem; color: #475569; margin-top: 4px; }}
  .thinking {{
    display: flex; gap: 4px; align-items: center; padding: 12px 16px;
    background: #1a1f2e; border-radius: 14px; border-top-left-radius: 4px;
  }}
  .dot {{
    width: 7px; height: 7px; border-radius: 50%; background: #475569;
    animation: bounce 1.2s infinite ease-in-out;
  }}
  .dot:nth-child(2) {{ animation-delay: .2s; }}
  .dot:nth-child(3) {{ animation-delay: .4s; }}
  @keyframes bounce {{
    0%, 80%, 100% {{ transform: scale(0.7); opacity: .5; }}
    40% {{ transform: scale(1); opacity: 1; }}
  }}
  footer {{
    padding: 16px 24px; border-top: 1px solid #1e2533; background: #141820;
  }}
  .input-row {{ display: flex; gap: 10px; max-width: 820px; margin: 0 auto; }}
  #input {{
    flex: 1; background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
    color: #e2e8f0; padding: 12px 16px; font-size: 0.92rem; resize: none;
    outline: none; line-height: 1.5; max-height: 140px; overflow-y: auto;
  }}
  #input:focus {{ border-color: #3b4fd8; }}
  #send {{
    background: #3b4fd8; color: #fff; border: none; border-radius: 10px;
    padding: 0 20px; font-size: 1.1rem; cursor: pointer; transition: background .15s;
    flex-shrink: 0;
  }}
  #send:hover {{ background: #4a5ee8; }}
  #send:disabled {{ background: #2d3748; cursor: not-allowed; }}
  .hint {{ text-align: center; font-size: 0.72rem; color: #334155; margin-top: 8px; }}
  /* Escalation banner */
  .escalation-banner {{
    background: #7c3a00; border-left: 3px solid #f97316;
    padding: 12px; border-radius: 6px; font-size: 0.85em;
    margin-top: 6px; color: #fed7aa;
  }}
  .escalation-banner .esc-reason {{
    display: block; font-size: 0.82em; color: #fdba74; margin-top: 2px;
  }}
  .escalation-banner .esc-score {{
    font-size: 0.78em; color: #fb923c; margin-top: 2px; display: block;
  }}
  .escalation-actions {{
    display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;
  }}
  .escalation-actions button {{
    font-size: 0.78em; padding: 4px 12px; border-radius: 6px;
    border: 1px solid #c2410c; background: #9a3412; color: #fff;
    cursor: pointer; transition: background .15s;
  }}
  .escalation-actions button:hover {{ background: #b45309; }}
  .escalation-preview {{
    background: #1a1010; border: 1px solid #7c3a00; border-radius: 6px;
    padding: 10px; margin-top: 10px; font-family: monospace;
    font-size: 0.78em; white-space: pre-wrap; color: #d1a87a;
  }}
  .claude-badge {{
    display: inline-block; background: #1e3a8a; color: #93c5fd;
    font-size: 0.7em; padding: 2px 8px; border-radius: 999px;
    margin-left: 6px; vertical-align: middle;
  }}
  .confidence-mini {{
    font-size: 0.68em; color: #475569; margin-top: 3px;
  }}
</style>
</head>
<body>
<header>
  <h1>&#128269; CodeIntelligence</h1>
  <div class="header-right">
    <div class="badges">
      <span class="badge green">&#9679; {chunks_count} chunks</span>
      <span class="badge">{backend}</span>
      <span class="badge">{retrieval} retrieval</span>
    </div>
    <button id="export-btn" disabled title="Export session">&#8595; Export session</button>
  </div>
</header>
<div id="chat">
  <div class="msg assistant">
    <div class="avatar">AI</div>
    <div>
      <div class="bubble">Hi! Ask me anything about this Python repository.
I have {chunks_count} semantic chunks indexed and ready.
Try something like: <em>"How does authentication work?"</em> or <em>"Where is the rate limiter implemented?"</em></div>
    </div>
  </div>
</div>
<footer>
  <div class="input-row">
    <textarea id="input" rows="1" placeholder="Ask a question about the codebase..."></textarea>
    <button id="send">&#8593;</button>
  </div>
  <div class="hint">Enter to send &nbsp;·&nbsp; Shift+Enter for new line</div>
</footer>
<script>
  const chat = document.getElementById('chat');
  const input = document.getElementById('input');
  const send = document.getElementById('send');

  function scrollBottom() {{
    chat.scrollTop = chat.scrollHeight;
  }}

  function addMessage(role, text) {{
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    const av = document.createElement('div');
    av.className = 'avatar';
    av.textContent = role === 'user' ? 'You' : 'AI';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(av);
    const inner = document.createElement('div');
    inner.appendChild(bubble);
    wrap.appendChild(inner);
    chat.appendChild(wrap);
    scrollBottom();
    return bubble;
  }}

  function addThinking() {{
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    wrap.id = 'thinking';
    const av = document.createElement('div');
    av.className = 'avatar';
    av.textContent = 'AI';
    const thinking = document.createElement('div');
    thinking.className = 'thinking';
    thinking.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    wrap.appendChild(av);
    const inner = document.createElement('div');
    inner.appendChild(thinking);
    wrap.appendChild(inner);
    chat.appendChild(wrap);
    scrollBottom();
    return wrap;
  }}

  async function ask(question) {{
    send.disabled = true;
    addMessage('user', question);
    const thinking = addThinking();

    try {{
      const res = await fetch('/v1/chat/completions', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          model: 'rag',
          messages: [{{ role: 'user', content: question }}],
          stream: true,
        }}),
      }});

      thinking.remove();
      const bubble = addMessage('assistant', '');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {{
        const {{ done, value }} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {{ stream: true }});
        const lines = buffer.split('\\n');
        buffer = lines.pop();
        for (const line of lines) {{
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') break;
          try {{
            const obj = JSON.parse(data);
            const token = obj.choices?.[0]?.delta?.content || '';
            if (token) {{ bubble.textContent += token; scrollBottom(); }}
          }} catch {{}}
        }}
      }}
    }} catch (e) {{
      thinking.remove();
      addMessage('assistant', 'Error: ' + e.message);
    }}

    send.disabled = false;
    input.focus();
  }}

  send.addEventListener('click', () => {{
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    input.style.height = 'auto';
    ask(q);
  }});

  input.addEventListener('keydown', e => {{
    if (e.key === 'Enter' && !e.shiftKey) {{
      e.preventDefault();
      send.click();
    }}
  }});

  input.addEventListener('input', () => {{
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  }});

  // ---- session history & export ----
  const SESSION = {{
    started_at: new Date().toISOString(),
    model: '{backend}',
    retrieval: '{retrieval}',
    chunks: {chunks_count},
    turns: [],
  }};

  const exportBtn = document.getElementById('export-btn');

  function recordTurn(question, answer, elapsed_ms) {{
    SESSION.turns.push({{
      timestamp: new Date().toISOString(),
      question,
      answer,
      elapsed_ms,
    }});
    exportBtn.disabled = false;
  }}

  function buildMarkdown() {{
    const lines = [
      '# CodeIntelligence — Session Report',
      '',
      `**Date:** ${{SESSION.started_at.replace('T',' ').slice(0,19)}} UTC`,
      `**Model:** ${{SESSION.model}}`,
      `**Retrieval:** ${{SESSION.retrieval}}`,
      `**Chunks indexed:** ${{SESSION.chunks}}`,
      `**Questions asked:** ${{SESSION.turns.length}}`,
      '',
      '---',
      '',
    ];
    SESSION.turns.forEach((t, i) => {{
      const secs = t.elapsed_ms != null ? (t.elapsed_ms / 1000).toFixed(1) + 's' : 'n/a';
      lines.push(`## Q${{i+1}} — ${{t.timestamp.replace('T',' ').slice(0,19)}} UTC`);
      lines.push('');
      lines.push(`**Question:** ${{t.question}}`);
      lines.push('');
      lines.push(`**Response time:** ${{secs}}`);
      lines.push('');
      lines.push(`**Answer:**`);
      lines.push('');
      lines.push(t.answer);
      lines.push('');
      lines.push('---');
      lines.push('');
    }});
    return lines.join('\\n');
  }}

  exportBtn.addEventListener('click', () => {{
    if (SESSION.turns.length === 0) return;

    // offer both Markdown and JSON
    const fmt = window.confirm(
      'Export as Markdown? (OK = .md, Cancel = .json)'
    );

    let content, filename, mime;
    if (fmt) {{
      content  = buildMarkdown();
      filename = `ci_session_${{new Date().toISOString().slice(0,10)}}.md`;
      mime     = 'text/markdown';
    }} else {{
      content  = JSON.stringify(SESSION, null, 2);
      filename = `ci_session_${{new Date().toISOString().slice(0,10)}}.json`;
      mime     = 'application/json';
    }}

    const blob = new Blob([content], {{ type: mime }});
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }});

  // patch ask() to record turns
  const _originalAsk = ask;
  window.ask = async function(question) {{
    // capture answer after streaming completes
    send.disabled = true;
    addMessage('user', question);
    const thinking = addThinking();
    let fullAnswer = '';
    const t0 = Date.now();

    // live timer shown inside the thinking indicator
    const timerSpan = document.createElement('span');
    timerSpan.style.cssText = 'margin-left:8px;font-size:0.78em;opacity:0.6;font-variant-numeric:tabular-nums;';
    thinking.appendChild(timerSpan);
    const timerInterval = setInterval(() => {{
      timerSpan.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
    }}, 100);

    try {{
      const res = await fetch('/v1/chat/completions', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          model: 'rag',
          messages: [{{ role: 'user', content: question }}],
          stream: true,
        }}),
      }});

      clearInterval(timerInterval);
      thinking.remove();
      const bubble = addMessage('assistant', '');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {{
        const {{ done, value }} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {{ stream: true }});
        const lines = buffer.split('\\n');
        buffer = lines.pop();
        for (const line of lines) {{
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') break;
          try {{
            const obj = JSON.parse(data);
            const token = obj.choices?.[0]?.delta?.content || '';
            if (token) {{ bubble.textContent += token; fullAnswer += token; scrollBottom(); }}
          }} catch {{}}
        }}
      }}

      // elapsed badge under the answer bubble
      const elapsed_ms = Date.now() - t0;
      const badge = document.createElement('div');
      badge.style.cssText = 'font-size:0.72em;opacity:0.45;margin-top:4px;text-align:right;';
      badge.textContent = '⏱ ' + (elapsed_ms / 1000).toFixed(1) + 's';
      bubble.parentNode.appendChild(badge);

      // -- confidence check (fire-and-forget in background) --
      checkEscalation(question, fullAnswer, bubble.parentNode);

    }} catch (e) {{
      clearInterval(timerInterval);
      thinking.remove();
      addMessage('assistant', 'Error: ' + e.message);
      fullAnswer = 'Error: ' + e.message;
    }}

    const elapsed_ms = Date.now() - t0;
    recordTurn(question, fullAnswer, elapsed_ms);
    send.disabled = false;
    input.focus();
  }};

  // ---- Escalation helpers ----

  async function checkEscalation(question, localAnswer, bubbleParent) {{
    try {{
      const res = await fetch('/query/escalate', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          question,
          local_answer: localAnswer,
          chunks_used_ids: [],
          confirm: false,
          redact_code: true,
        }}),
      }});
      if (!res.ok) return;
      const data = await res.json();

      // Always show a mini confidence note
      const miniScore = document.createElement('div');
      miniScore.className = 'confidence-mini';
      miniScore.textContent = 'Confidence score: ' + (data.confidence_score * 100).toFixed(0) + '%';
      bubbleParent.appendChild(miniScore);

      if (!data.should_escalate) return;

      if (!data.escalation_available) {{
        // Server has no API key -- show only the mini note
        miniScore.textContent += '  ⚠ Low confidence — start server with --claude-api-key to enable escalation';
        return;
      }}

      // Build the escalation banner
      const banner = document.createElement('div');
      banner.className = 'escalation-banner';
      banner.innerHTML =
        '⚠ Low confidence response (score: ' + (data.confidence_score * 100).toFixed(0) + '%)' +
        '<span class="esc-reason">' + (data.reason || '') + '</span>' +
        '<div class="escalation-actions">' +
          '<button class="btn-preview">Show what would be sent</button>' +
          '<button class="btn-escalate">Use Claude API</button>' +
          '<button class="btn-dismiss">Dismiss</button>' +
        '</div>';
      bubbleParent.appendChild(banner);

      // Preview button
      banner.querySelector('.btn-preview').addEventListener('click', async () => {{
        // Toggle preview panel
        let panel = banner.querySelector('.escalation-preview');
        if (panel) {{ panel.remove(); return; }}

        const pres = await fetch('/query/escalate', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            question,
            local_answer: localAnswer,
            chunks_used_ids: [],
            confirm: false,
            redact_code: true,
          }}),
        }});
        const pdata = await pres.json();
        panel = document.createElement('div');
        panel.className = 'escalation-preview';
        panel.textContent = pdata.preview || '(no preview)';
        banner.appendChild(panel);
      }});

      // Use Claude API button
      banner.querySelector('.btn-escalate').addEventListener('click', async () => {{
        banner.querySelector('.btn-escalate').disabled = true;
        banner.querySelector('.btn-escalate').textContent = 'Calling Claude...';
        try {{
          const cres = await fetch('/query/escalate', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
              question,
              local_answer: localAnswer,
              chunks_used_ids: [],
              confirm: true,
              redact_code: true,
            }}),
          }});
          const cdata = await cres.json();
          if (cdata.error) {{
            alert('Escalation error: ' + cdata.error);
            return;
          }}
          // Show Claude answer as a new bubble
          const wrap = document.createElement('div');
          wrap.className = 'msg assistant';
          const av = document.createElement('div');
          av.className = 'avatar';
          av.textContent = 'AI';
          const inner = document.createElement('div');
          const claudeBubble = document.createElement('div');
          claudeBubble.className = 'bubble';
          claudeBubble.textContent = cdata.answer || '';
          const claudeBadge = document.createElement('span');
          claudeBadge.className = 'claude-badge';
          claudeBadge.textContent = 'Claude API';
          inner.appendChild(claudeBubble);
          inner.appendChild(claudeBadge);
          wrap.appendChild(av);
          wrap.appendChild(inner);
          chat.appendChild(wrap);
          chat.scrollTop = chat.scrollHeight;
          banner.remove();
          // Record escalated answer in session export
          recordTurn('[Escalated] ' + question, '[Local] ' + localAnswer + '\\n\\n[Claude API] ' + (cdata.answer || ''), null);
        }} catch (err) {{
          alert('Escalation failed: ' + err.message);
          banner.querySelector('.btn-escalate').disabled = false;
          banner.querySelector('.btn-escalate').textContent = 'Use Claude API';
        }}
      }});

      // Dismiss button
      banner.querySelector('.btn-dismiss').addEventListener('click', () => {{
        banner.remove();
      }});

    }} catch (_) {{
      // Silent failure -- escalation check is best-effort
    }}
  }}

  // override the send handler to use the new ask
  send.onclick = () => {{
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    input.style.height = 'auto';
    window.ask(q);
  }};
</script>
</body>
</html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self._read_body()
        if self.path == "/v1/chat/completions":
            self._openai_chat(body)
        elif self.path == "/v1/messages":
            self._anthropic_messages(body)
        elif self.path == "/api/chat":
            self._ollama_chat(body)
        elif self.path == "/api/generate":
            self._ollama_generate(body)
        elif self.path == "/query":
            self._simple_query(body)
        elif self.path == "/query/escalate":
            self._escalate_query(body)
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
            chunk = {
                "id": comp_id, "object": "chat.completion.chunk",
                "created": created, "model": "rag",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            self._write(f"data: {json.dumps(chunk)}\n\n")
            for item in _chat_stream(ollama_msgs):
                if isinstance(item, str):
                    token = item  # Claude yields str tokens directly
                else:
                    line = item.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("message", {}).get("content", "")
                    if obj.get("done"):
                        break
                if token:
                    chunk = {
                        "id": comp_id, "object": "chat.completion.chunk",
                        "created": created, "model": "rag",
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                    self._write(f"data: {json.dumps(chunk)}\n\n")
            chunk = {
                "id": comp_id, "object": "chat.completion.chunk",
                "created": created, "model": "rag",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            self._write(f"data: {json.dumps(chunk)}\n\n")
            self._write("data: [DONE]\n\n")
        else:
            answer = _chat_complete(ollama_msgs)
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
            for item in _chat_stream(ollama_msgs):
                if isinstance(item, str):
                    token = item
                    done = False
                else:
                    line = item.decode("utf-8").strip()
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
            answer = _chat_complete(ollama_msgs)
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

    # ---- Anthropic Messages API endpoint ----

    def _anthropic_messages(self, body: dict):
        """
        POST /v1/messages — native Anthropic Messages API format.

        Compatible with:
            client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="rag")
            client.messages.create(model="rag", max_tokens=1024, messages=[...])
        """
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if not question:
            self._json({"type": "error", "error": {"type": "invalid_request_error", "message": "no user message"}}, 400)
            return

        augmented_msgs, relevant = build_messages(question, messages)
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            # Anthropic SSE streaming format
            self._sse_start()

            def sse(event: str, data: dict):
                self._write(f"event: {event}\ndata: {json.dumps(data)}\n\n")

            sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "model": CONFIG["model"], "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            })
            sse("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
            sse("ping", {"type": "ping"})

            for item in _chat_stream(augmented_msgs):
                if isinstance(item, str):
                    token = item
                else:
                    line = item.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("message", {}).get("content", "")
                    if obj.get("done"):
                        break
                if token:
                    sse("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": token},
                    })

            sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": -1},
            })
            sse("message_stop", {"type": "message_stop"})
        else:
            answer = _chat_complete(augmented_msgs)
            self._json({
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": CONFIG["model"],
                "content": [{"type": "text", "text": answer}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": -1, "output_tokens": -1},
                "_rag_chunks_used": len(relevant),
            })

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
        answer = _chat_complete(ollama_msgs)
        self._json({
            "answer": answer,
            "sources": [
                {"id": c["id"], "type": c["type"],
                 "content": c["text"][:300] + ("..." if len(c["text"]) > 300 else "")}
                for c in relevant
            ],
        })

    # ---- Escalation endpoint ----

    def _escalate_query(self, body: dict):
        """
        POST /query/escalate

        Request body:
            {
                "question":        str,
                "local_answer":    str,
                "chunks_used_ids": list[str],   # empty => use LAST_CHUNKS
                "confirm":         bool,         # false => preview only
                "redact_code":     bool          # true by default
            }

        If confirm=false  -> returns preview + confidence info (no Claude call).
        If confirm=true   -> calls Claude API and returns the upgraded answer.
        """
        from escalation import (
            score_confidence,
            build_escalation_payload,
            escalate_to_claude,
        )

        question = body.get("question", "")
        local_answer = body.get("local_answer", "")
        chunks_used_ids: list[str] = body.get("chunks_used_ids", [])
        confirm: bool = body.get("confirm", False)
        redact_code: bool = body.get("redact_code", True)

        if not question:
            self._json({"error": "missing question"}, 400)
            return

        # Resolve chunks
        if chunks_used_ids:
            chunk_by_id = {c["id"]: c for c in CHUNKS}
            chunks_for_escalation = [
                chunk_by_id[cid] for cid in chunks_used_ids if cid in chunk_by_id
            ]
        else:
            chunks_for_escalation = list(LAST_CHUNKS)

        # Score confidence
        threshold = CONFIG.get("escalation_threshold", 0.35)
        conf = score_confidence(local_answer, question, chunks_for_escalation)

        if not confirm:
            # Preview mode -- never call Claude
            esc_api_key = CONFIG.get("escalation_api_key", "")
            if not esc_api_key:
                # Server has no key -- return a minimal response
                self._json({
                    "should_escalate": conf["should_escalate"],
                    "confidence_score": conf["score"],
                    "signals": conf["signals"],
                    "reason": conf["reason"],
                    "escalation_available": False,
                    "message": (
                        "Start server with --claude-api-key to enable escalation"
                    ),
                })
                return

            esc = build_escalation_payload(
                question=question,
                chunks=chunks_for_escalation,
                local_answer=local_answer,
                redact_code=redact_code,
            )
            cost_usd = esc["estimated_tokens"] * 0.000005
            self._json({
                "should_escalate": conf["should_escalate"],
                "confidence_score": conf["score"],
                "signals": conf["signals"],
                "reason": conf["reason"],
                "preview": esc["preview"],
                "estimated_tokens": esc["estimated_tokens"],
                "estimated_cost_usd": round(cost_usd, 6),
                "redacted": esc["redacted"],
                "escalation_available": True,
                "action_required": "Set confirm:true to proceed",
            })
            return

        # confirm=True -- call Claude (requires api key)
        esc_api_key = CONFIG.get("escalation_api_key", "")
        if not esc_api_key:
            self._json({
                "error": (
                    "Escalation API key not configured. "
                    "Start server with --claude-api-key <key> or set ANTHROPIC_API_KEY."
                )
            }, 503)
            return

        esc = build_escalation_payload(
            question=question,
            chunks=chunks_for_escalation,
            local_answer=local_answer,
            redact_code=redact_code,
        )

        try:
            claude_answer = escalate_to_claude(esc["payload"], esc_api_key)
        except (ValueError, ConnectionError) as exc:
            self._json({"error": str(exc)}, 502)
            return

        self._json({
            "source": "claude-api",
            "model": "claude-opus-4-7",
            "answer": claude_answer,
            "local_answer": local_answer,
            "confidence_score": conf["score"],
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

    port                 = int(_pop_arg(args, "--port")                or 8080)
    model                = _pop_arg(args, "--model")                   or "qwen2.5-coder:7b"
    ollama_url           = _pop_arg(args, "--ollama")                  or "http://localhost:11434"
    top_k                = int(_pop_arg(args, "--top-k")               or 6)
    embed_model          = _pop_arg(args, "--embed-model")             or "nomic-embed-text"
    api_key              = _pop_arg(args, "--api-key")                 or ""
    chroma_dir           = _pop_arg(args, "--chroma-dir")              or None
    escalation_threshold = float(_pop_arg(args, "--escalation-threshold") or 0.35)
    claude_api_key_arg   = _pop_arg(args, "--claude-api-key")          or ""
    use_embed   = "--embed"   in args
    use_claude  = "--claude"  in args
    use_rerank  = "--rerank"  in args
    use_chroma  = "--chroma"  in args
    if use_embed:  args.remove("--embed")
    if use_claude: args.remove("--claude")
    if use_rerank: args.remove("--rerank")
    if use_chroma: args.remove("--chroma")

    if use_chroma and not use_embed:
        print("Error: --chroma requires --embed (semantic mode)", file=sys.stderr)
        sys.exit(1)

    import os
    if use_claude:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("Error: --claude requires --api-key or ANTHROPIC_API_KEY env var", file=sys.stderr)
            sys.exit(1)
        model = model if model != "qwen2.5-coder:7b" else "claude-opus-4-7"

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

    # Initialise ChromaDB collection if requested
    chroma_collection = None
    if use_chroma:
        try:
            from chroma_store import build_chroma_index  # type: ignore
            chroma_collection = build_chroma_index(
                str(jsonl_path),
                persist_dir=chroma_dir,
            )
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Error initialising ChromaDB: {exc}", file=sys.stderr)
            sys.exit(1)

    # Resolve escalation API key: explicit arg > env var
    escalation_api_key = (
        claude_api_key_arg
        or os.environ.get("ANTHROPIC_API_KEY", "")
        # If --claude flag is set the main api_key already came from env
    )
    # But don't duplicate the primary claude key into the escalation slot
    # unless it was explicitly supplied via --claude-api-key.
    if not escalation_api_key and use_claude:
        # When running with --claude the primary api_key IS the Anthropic key
        escalation_api_key = api_key

    CONFIG = {
        "model": model,
        "ollama_url": ollama_url,
        "top_k": top_k,
        "use_embed": use_embed,
        "embed_model": embed_model,
        "use_claude": use_claude,
        "claude_api_key": api_key,
        "use_rerank": use_rerank,
        "use_chroma": use_chroma,
        "chroma_collection": chroma_collection,
        "escalation_threshold": escalation_threshold,
        "escalation_api_key": escalation_api_key,
    }

    retrieval = f"semantic ({embed_model})" if use_embed else "TF-IDF"
    rerank_label = " + cross-encoder rerank" if use_rerank else ""
    vector_store_label = " [ChromaDB]" if use_chroma else " [numpy in-memory]"
    backend = f"Claude API ({model})" if use_claude else f"Ollama ({model} at {ollama_url})"

    print(f"RAG Server starting...")
    print(f"  Chunks     : {len(CHUNKS)} from {jsonl_path.name}")
    print(f"  Backend    : {backend}")
    print(f"  Retrieval  : {retrieval}{rerank_label}{vector_store_label}  |  top_k: {top_k}")
    print(f"  Port       : {port}")
    print()
    escalation_label = (
        f"enabled (threshold={escalation_threshold})"
        if escalation_api_key else "disabled (no --claude-api-key)"
    )
    print(f"  OpenAI endpoint   : http://localhost:{port}/v1/chat/completions")
    print(f"  Anthropic endpoint: http://localhost:{port}/v1/messages")
    print(f"  Ollama endpoint   : http://localhost:{port}/api/chat")
    print(f"  Health check      : http://localhost:{port}/health")
    print(f"  Escalation        : http://localhost:{port}/query/escalate  [{escalation_label}]")
    print()
    print(f"  Connect Open WebUI      -> Ollama URL: http://localhost:{port}")
    print(f"  Connect OpenAI client   -> base_url=http://localhost:{port}/v1   api_key=rag")
    print(f"  Connect Anthropic SDK   -> base_url=http://localhost:{port}      api_key=rag")
    print()
    print("Press Ctrl+C to stop.\n")

    server = HTTPServer(("0.0.0.0", port), RAGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
