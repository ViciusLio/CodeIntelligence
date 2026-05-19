# CodeIntelligence — RAG toolkit for Python repositories

> **Query any Python codebase in natural language — fully local, zero API key, zero data leaving your machine.**

---

## What is this?

CodeIntelligence is a lightweight RAG (Retrieval-Augmented Generation) toolkit that makes any Python repository queryable by an AI model.

Instead of pasting code into a chat or hoping the model knows your codebase, you:

1. **Parse** the repo into semantic chunks (one per file, function, class)
2. **Retrieve** the most relevant chunks for your question
3. **Generate** a grounded answer via a local LLM (Ollama) or Claude API

The result: ask questions like *"Where is JWT authentication implemented?"* or *"How does the rate limiter work?"* and get precise, source-cited answers — without the model having to read 10,000 lines of code every time.

---

## Local or Cloud — your choice

The chunk format is model-agnostic. Once you have `repo_chunks.jsonl` you can query it
with a local LLM **or** Claude API — same commands, same results.

| | Local (Ollama) | Cloud (Claude API) |
|--|----------------|--------------------|
| **Script** | `ask_repo_local.py` / `rag_server.py` | `ask_repo.py` |
| **Model** | qwen2.5-coder, codellama, llama3… | claude-opus-4-7 |
| **API key** | not required | `ANTHROPIC_API_KEY` |
| **Data privacy** | stays on your machine | sent to Anthropic |
| **Quality** | good (7B–13B models) | excellent |
| **Cost** | free | pay per token |
| **Best for** | internal / sensitive repos | highest quality answers |

```bash
# Local — Ollama
python ask_repo_local.py repo_chunks.jsonl "How does authentication work?" \
    --model qwen2.5-coder:7b --top-k 4 --no-stream

# Cloud — Claude API
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python ask_repo.py repo_chunks.jsonl "How does authentication work?" --top-k 8
```

Both read the same `repo_chunks.jsonl` — no need to re-parse.

---

## Repository structure

```
CodeIntelligence/
├── parse_repo.py        # Step 1 — AST parser: repo → repo_chunks.jsonl
├── embed_chunks.py      # Step 2 (optional) — enrich chunks with Ollama embeddings
├── ask_repo.py          # Query via Claude API  (anthropic SDK)
├── ask_repo_local.py    # Query via Ollama      (stdlib only)
└── rag_server.py        # HTTP server — OpenAI + Anthropic + Ollama endpoints + chat UI
```

---

## Quick start (5 minutes)

### Prerequisites

```bash
# 1. Install Ollama from https://ollama.com
# 2. Pull a model (recommended for code)
ollama pull qwen2.5-coder:7b

# Ollama starts automatically — verify it's running
curl http://localhost:11434/
# → Ollama is running
```

### Step 1 — Parse your repo

```bash
python parse_repo.py /path/to/your/repo --output repo_chunks.jsonl
```

This reads every `.py` file using Python's `ast` module and produces a JSONL file
with one chunk per semantic unit:

| Chunk type | Content |
|------------|---------|
| `repo_overview` | folder structure, entry points, top imports |
| `file::<path>` | imports, exports, module docstring |
| `function::<file>::<name>` | signature, docstring, full body, called functions |
| `class::<file>::<name>` | methods, attributes, inheritance |

Example output for a medium repo (~12k lines):
```
Found 122 Python files
Wrote 1,561 chunks → repo_chunks.jsonl
```

### Step 2 — Start the server

```bash
python rag_server.py repo_chunks.jsonl --port 8080 --model qwen2.5-coder:7b
```

Then open **http://localhost:8080** — a built-in chat UI is ready.

---

## Built-in Chat UI

Open `http://localhost:8080` in any browser after starting the server.

- Dark theme chat with streaming tokens visible as they arrive
- "Thinking..." animation while the model is working
- Badges showing model, backend, and chunk count
- Enter to send · Shift+Enter for new line
- No frontend to install — served directly by `rag_server.py`

---

## Retrieval modes — TF-IDF vs Semantic

By default the server uses TF-IDF (keyword matching). For significantly better results
on questions involving synonyms, acronyms, or concept-level reasoning, switch to
**semantic retrieval** using Ollama embeddings.

### The difference in practice

> Question: *"Where is JWT token creation implemented?"*

| | TF-IDF | Semantic |
|--|--------|----------|
| File found | ❌ hallucinated `app/utils/jwt.py` | ✅ `app/core/security.py` |
| Answer | invented | based on real code |
| Claims listed | generic | exact: `sub`, `iat`, `exp`, `type`, `jti` |

TF-IDF matches the word "JWT" literally — it misses `create_access_token` in `security.py`
because the word "JWT" doesn't appear enough in that chunk.
Semantic embeddings understand that *"JWT token creation"* and *"create_access_token"*
refer to the same concept.

### Setup semantic retrieval

```bash
# 1. Pull the embedding model (~270 MB, very fast)
ollama pull nomic-embed-text

# 2. Enrich your chunks with embeddings (one-off, ~2-3 min for a 12k-line repo)
python embed_chunks.py repo_chunks.jsonl
# → writes repo_chunks_embedded.jsonl

# 3. Start the server with semantic retrieval
python rag_server.py repo_chunks_embedded.jsonl --port 8080 --embed --model qwen2.5-coder:7b
```

### How embed_chunks.py works

```
repo_chunks.jsonl (text only)
        │
        ▼  embed_chunks.py calls Ollama /api/embeddings for each chunk
        │  (nomic-embed-text → 768-dimensional vector per chunk)
        ▼
repo_chunks_embedded.jsonl  (same chunks + "embedding": [...] field)
        │
        ▼  at query time: embed the question → cosine similarity → top-k
```

Supports `--resume` to safely continue an interrupted embedding run.

---

## Serving the API — connect any frontend

`rag_server.py` exposes **four API formats simultaneously**:

```bash
python rag_server.py repo_chunks.jsonl --port 8080 --model qwen2.5-coder:7b
```

```
  Chat UI           : http://localhost:8080/
  OpenAI endpoint   : http://localhost:8080/v1/chat/completions
  Anthropic endpoint: http://localhost:8080/v1/messages
  Ollama endpoint   : http://localhost:8080/api/chat
  Health check      : http://localhost:8080/health
```

### Connect your frontend

| Frontend | Configuration |
|----------|---------------|
| **Browser** | open `http://localhost:8080` |
| **Open WebUI** | Settings → Connections → Ollama URL: `http://localhost:8080` |
| **LibreChat** | `OLLAMA_BASE_URL=http://localhost:8080` |
| **Chatbox** | Provider: Ollama → URL: `http://localhost:8080` |
| **OpenAI SDK** | `base_url="http://localhost:8080/v1"`, `api_key="rag"` |
| **Anthropic SDK** | `base_url="http://localhost:8080"`, `api_key="rag"` |

### Connect via SDK

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="rag")
response = client.chat.completions.create(
    model="rag",
    messages=[{"role": "user", "content": "Where is JWT implemented?"}]
)

# Anthropic SDK
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="rag")
response = client.messages.create(
    model="rag", max_tokens=1024,
    messages=[{"role": "user", "content": "Where is JWT implemented?"}]
)
```

### Test with curl

```bash
# Health check
curl http://localhost:8080/health

# Simple query (easiest)
curl http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is JWT token creation implemented?"}'

# OpenAI format
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rag","messages":[{"role":"user","content":"How does rate limiting work?"}],"stream":false}'

# Anthropic format
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"rag","max_tokens":1024,"messages":[{"role":"user","content":"How does rate limiting work?"}]}'
```

---

## Using Claude API as backend

You can replace Ollama with Claude as the generation backend — same endpoints,
same UI, better quality answers.

```bash
# With --claude flag (uses claude-opus-4-7 by default)
python rag_server.py repo_chunks.jsonl --port 8080 --claude --api-key sk-...

# Or via environment variable
export ANTHROPIC_API_KEY=sk-...
python rag_server.py repo_chunks.jsonl --port 8080 --claude
```

The health check tells you which backend is active:
```json
{"status": "ok", "chunks": 1561, "model": "claude-opus-4-7", "backend": "claude", "retrieval": "tfidf"}
```

---

## All options

### parse_repo.py
```
python parse_repo.py <repo_path> [--output chunks.jsonl] [--exclude <pattern>]

  --output   Output JSONL file      (default: repo_chunks.jsonl)
  --exclude  Skip paths matching pattern (repeatable)
             e.g. --exclude tests --exclude migrations
```

### embed_chunks.py
```
python embed_chunks.py <chunks.jsonl> [options]

  --output       Output file          (default: <input>_embedded.jsonl)
  --model        Ollama embedding model (default: nomic-embed-text)
  --url          Ollama base URL      (default: http://localhost:11434)
  --resume       Skip already-embedded chunks (safe to re-run after interruption)
```

### ask_repo_local.py
```
python ask_repo_local.py <chunks.jsonl> ["question"] [options]

  --model        Ollama LLM model     (default: codellama)
  --top-k        Chunks to retrieve   (default: 12)
  --embed        Use semantic retrieval (needs embedded JSONL)
  --embed-model  Embedding model      (default: nomic-embed-text)
  --no-stream    Print full answer at once
  --all-chunks   Send all chunks as context
```

### rag_server.py
```
python rag_server.py <chunks.jsonl> [options]

  --port         Listening port       (default: 8080)
  --model        Ollama LLM model     (default: qwen2.5-coder:7b)
  --top-k        Chunks per query     (default: 6)
  --embed        Use semantic retrieval (needs embedded JSONL)
  --embed-model  Embedding model      (default: nomic-embed-text)
  --ollama       Ollama base URL      (default: http://localhost:11434)
  --claude       Use Claude API as backend instead of Ollama
  --api-key      Anthropic API key    (or set ANTHROPIC_API_KEY)
```

---

## Benchmark repos

This toolkit was built alongside three purpose-made benchmark repositories
for evaluating RAG pipelines on Python code:

| Repo | Size | Domain | Chunks |
|------|------|--------|--------|
| [ci-bench-L1](https://github.com/ViciusLio/ci-bench-L1) | ~6k lines | Validation library | 584 |
| [ci-bench-L2](https://github.com/ViciusLio/ci-bench-L2) | ~12k lines | FastAPI REST service | 1,561 |
| [ci-bench-L3](https://github.com/ViciusLio/ci-bench-L3) | ~18k lines | Pipeline framework | 2,977 |

Each repo includes ground truth Q&A pairs and evaluation metrics (Hit@K, MRR, MAP, NDCG)
so you can measure exactly how well your pipeline performs.

```bash
# Full end-to-end example with ci-bench-L2
git clone https://github.com/ViciusLio/ci-bench-L2

# Parse
python parse_repo.py ../ci-bench-L2 --output ci_bench_L2_chunks.jsonl

# Embed (recommended)
ollama pull nomic-embed-text
python embed_chunks.py ci_bench_L2_chunks.jsonl

# Start server with semantic retrieval
python rag_server.py ci_bench_L2_chunks_embedded.jsonl --port 8080 --embed

# Open the chat UI
open http://localhost:8080

# Run the benchmark evaluation
cd ../ci-bench-L2
python benchmarks/eval_harness.py --rag-url http://localhost:8080 --output results.json
```

---

## Advanced

### Cross-encoder reranking (`--rerank`)

Retrieval quality can be improved by adding a **cross-encoder reranker** as a second stage.
Instead of comparing a query embedding against chunk embeddings in isolation, the cross-encoder
reads every (query, chunk) pair together and scores their relevance — much more accurate but
also more CPU-intensive.

```bash
# Install the dependency (~80 MB, CPU-friendly)
pip install sentence-transformers

# Use reranking with semantic retrieval (recommended)
python ask_repo_local.py repo_chunks_embedded.jsonl "Where is JWT created?" \
    --embed --rerank --top-k 5

# Use reranking with TF-IDF retrieval
python ask_repo_local.py repo_chunks.jsonl "Where is JWT created?" \
    --rerank --top-k 5

# Enable on the server
python rag_server.py repo_chunks_embedded.jsonl --embed --rerank
```

When `--rerank` is active, the pipeline retrieves `top_k * 3` candidates via the initial
retrieval method, then re-scores them with `cross-encoder/ms-marco-MiniLM-L-6-v2` and
returns the best `top_k`.

If `sentence-transformers` is not installed, the flag is silently ignored and the original
ranking is used — no crash, no noise.

The health endpoint reflects the active reranker:
```json
{"status": "ok", "reranker": "cross-encoder", "vector_store": "numpy_in_memory", ...}
```

---

### ChromaDB persistent vector store (`--chroma`)

By default semantic retrieval loads all embeddings into memory on every server start.
For large repos (millions of embeddings) you can use **ChromaDB** as a persistent vector store.

```bash
# Install ChromaDB
pip install chromadb

# Build the index once (idempotent — safe to re-run)
python chroma_store.py build repo_chunks_embedded.jsonl

# Query the index directly (uses Ollama to embed the question)
python chroma_store.py query repo_chunks_embedded.jsonl "how does auth work" --top-k 5

# Show collection info
python chroma_store.py info repo_chunks_embedded.jsonl

# Start the server with ChromaDB (--chroma requires --embed)
python rag_server.py repo_chunks_embedded.jsonl --embed --chroma
```

ChromaDB persists data in `.codeintelligence_chroma/` in the current directory.
Multiple repos coexist in the same Chroma instance because each gets its own
collection named after the JSONL file stem.

```bash
# Custom persist directory
python rag_server.py repo_chunks_embedded.jsonl --embed --chroma --chroma-dir /data/chroma
```

The health endpoint reflects the active store:
```json
{"status": "ok", "vector_store": "chroma", "reranker": "none", ...}
```

---

### Usages / call-graph chunks (parse_repo.py)

By default `parse_repo.py` now also emits `usages` chunks — cross-file call graph entries
that let the RAG system answer questions like *"Who calls `create_access_token`?"*.

```bash
# Default — includes usages chunks
python parse_repo.py /my/repo --output chunks.jsonl

# Opt-out if you want the old behaviour (no usages chunks)
python parse_repo.py /my/repo --output chunks.jsonl --no-usages
```

---

### Multi-language parsing (`--include-langs`)

By default only Python files are parsed.  Pass `--include-langs` to also parse
Markdown, YAML/TOML config files, and JavaScript/TypeScript files.

```bash
# Parse everything
python parse_repo.py /my/repo \
    --include-langs markdown config js \
    --output chunks.jsonl

# Only add Markdown docs
python parse_repo.py /my/repo --include-langs markdown --output chunks.jsonl
```

| Lang key | Extensions | Chunk type |
|----------|------------|------------|
| `markdown` | `.md`, `.mdx` | `doc` — one chunk per H1/H2 section |
| `config` | `.yml`, `.yaml`, `.toml` | `config` — one chunk per file with top-level keys |
| `js` | `.js`, `.ts`, `.jsx`, `.tsx` | `file_js` — exports, imports, classes |

---

### Incremental embedding (`--incremental`)

After the initial embedding run, re-embedding only the files that changed saves time on
large repos.  The tool stores a sidecar `<output>_file_hashes.json` with the SHA-256
hash of every source file.

```bash
# First run — embeds everything
python embed_chunks.py repo_chunks.jsonl

# After editing a few files — only re-embeds changed files
python embed_chunks.py repo_chunks.jsonl --incremental

# Output:
#   3 files unchanged (embeddings reused)
#   1 files changed/new (re-embedded)
#   1561 chunks total
```

`--incremental` and `--resume` are independent flags and can be combined.

### All new flags summary

#### parse_repo.py
```
  --no-usages            Skip emitting usages/call-graph chunks
  --include-langs <l>    Also parse: markdown  config  js  (space-separated)
```

#### embed_chunks.py
```
  --incremental          Skip re-embedding unchanged source files (uses SHA-256 hashing)
```

#### ask_repo_local.py
```
  --rerank               Re-score retrieved chunks with a cross-encoder (sentence-transformers)
```

#### rag_server.py
```
  --rerank               Re-score retrieved chunks with a cross-encoder
  --chroma               Use ChromaDB as persistent vector store (requires --embed)
  --chroma-dir <dir>     ChromaDB persistence directory (default: .codeintelligence_chroma/)
```

---

## License

MIT — free to use for internal tooling, research, and commercial evaluation.
