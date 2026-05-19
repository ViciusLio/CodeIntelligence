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

## Why local?

- **No data leaves your machine** — works entirely on localhost
- **No API key needed** — uses Ollama with any open model
- **No external dependencies** for the parser — pure Python stdlib (`ast`, `pathlib`, `json`)
- **Model-agnostic** — the JSONL chunk format works with any LLM

---

## Repository structure

```
CodeIntelligence/
├── parse_repo.py        # Step 1 — AST parser: repo → repo_chunks.jsonl
├── embed_chunks.py      # Step 2 (optional) — enrich chunks with Ollama embeddings
├── ask_repo.py          # Query via Claude API  (anthropic SDK)
├── ask_repo_local.py    # Query via Ollama      (stdlib only)
└── rag_server.py        # HTTP server — exposes OpenAI + Ollama compatible endpoints
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

### Step 2 — Ask questions

**Option A — interactive CLI (Ollama)**
```bash
python ask_repo_local.py repo_chunks.jsonl --model qwen2.5-coder:7b --top-k 4 --no-stream
# → interactive mode, type your questions
```

**Option B — single question**
```bash
python ask_repo_local.py repo_chunks.jsonl "How does the authentication flow work?" \
    --model qwen2.5-coder:7b --top-k 4 --no-stream
```

**Option C — expose as API and connect any frontend**
```bash
python rag_server.py repo_chunks.jsonl --port 8080 --model qwen2.5-coder:7b
```

---

## Serving the API — connect any frontend

`rag_server.py` starts an HTTP server that speaks both **OpenAI** and **Ollama** protocols
simultaneously — no frontend changes needed.

```bash
python rag_server.py repo_chunks.jsonl --port 8080 --model qwen2.5-coder:7b
```

```
RAG Server starting...
  Chunks     : 1561 from repo_chunks.jsonl
  LLM model  : qwen2.5-coder:7b  (via Ollama at http://localhost:11434)
  Retrieval  : TF-IDF  |  top_k: 6

  OpenAI endpoint : http://localhost:8080/v1/chat/completions
  Ollama endpoint : http://localhost:8080/api/chat
  Health check    : http://localhost:8080/health
```

### Connect your frontend

| Frontend | Configuration |
|----------|---------------|
| **Open WebUI** | Settings → Connections → Ollama URL: `http://localhost:8080` |
| **LibreChat** | `OLLAMA_BASE_URL=http://localhost:8080` |
| **Chatbox** | Provider: Ollama → URL: `http://localhost:8080` |
| **Any OpenAI SDK/client** | `base_url="http://localhost:8080/v1"`, `api_key="rag"` |

### Test with curl

```bash
# OpenAI format
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rag","messages":[{"role":"user","content":"Where is JWT implemented?"}],"stream":false}'

# Ollama format
curl http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"rag","messages":[{"role":"user","content":"Where is JWT implemented?"}],"stream":false}'

# Simple query endpoint
curl http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does the rate limiter work?"}'
```

---

## Improving retrieval quality — semantic embeddings

By default, retrieval uses TF-IDF (keyword matching). For better results on complex
questions, you can switch to semantic retrieval using Ollama embeddings.

```bash
# 1. Pull the embedding model (~270 MB, very fast)
ollama pull nomic-embed-text

# 2. Enrich your chunks with embeddings (one-off, ~2-3 min for a 12k-line repo)
python embed_chunks.py repo_chunks.jsonl
# → writes repo_chunks_embedded.jsonl

# 3. Use semantic retrieval
python ask_repo_local.py repo_chunks_embedded.jsonl "Where is JWT implemented?" \
    --embed --model qwen2.5-coder:7b --top-k 4 --no-stream

# Or via the server
python rag_server.py repo_chunks_embedded.jsonl --embed --port 8080
```

### TF-IDF vs Semantic

| | TF-IDF | Semantic (nomic-embed-text) |
|--|--------|-----------------------------|
| Setup | none | `ollama pull nomic-embed-text` + `embed_chunks.py` |
| Query latency | < 1ms | ~50ms (embed query + cosine similarity) |
| Finds exact keywords | yes | yes |
| Finds synonyms / concepts | no | **yes** |
| Example: "JWT" finds `create_access_token` | no | **yes** |

---

## All options

### parse_repo.py
```
python parse_repo.py <repo_path> [--output chunks.jsonl] [--exclude <pattern>]

  --output   Output JSONL file     (default: repo_chunks.jsonl)
  --exclude  Skip paths matching pattern (repeatable, e.g. --exclude tests --exclude migrations)
```

### embed_chunks.py
```
python embed_chunks.py <chunks.jsonl> [--output embedded.jsonl] [--model nomic-embed-text] [--resume]

  --output   Output file           (default: <input>_embedded.jsonl)
  --model    Ollama embedding model (default: nomic-embed-text)
  --resume   Skip already-embedded chunks (safe to re-run after interruption)
```

### ask_repo_local.py
```
python ask_repo_local.py <chunks.jsonl> ["question"] [options]

  --model        Ollama LLM model    (default: codellama)
  --top-k        Chunks to retrieve  (default: 12)
  --embed        Use semantic retrieval
  --embed-model  Embedding model     (default: nomic-embed-text)
  --no-stream    Print full answer at once instead of streaming
  --all-chunks   Send all chunks as context
```

### rag_server.py
```
python rag_server.py <chunks.jsonl> [options]

  --port         Listening port      (default: 8080)
  --model        Ollama LLM model    (default: qwen2.5-coder:7b)
  --top-k        Chunks per query    (default: 6)
  --embed        Use semantic retrieval
  --embed-model  Embedding model     (default: nomic-embed-text)
  --ollama       Ollama base URL     (default: http://localhost:11434)
```

---

## Use with Claude API

If you prefer Claude over a local model, `ask_repo.py` uses the Anthropic SDK
with `claude-opus-4-7` and adaptive thinking:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...

python ask_repo.py repo_chunks.jsonl "Explain the permission system" --top-k 8
```

---

## Benchmark repos

This toolkit was built alongside three purpose-made benchmark repositories
for evaluating RAG pipelines on Python code:

| Repo | Size | Domain |
|------|------|--------|
| [ci-bench-L1](https://github.com/ViciusLio/ci-bench-L1) | ~6k lines | Validation library |
| [ci-bench-L2](https://github.com/ViciusLio/ci-bench-L2) | ~12k lines | FastAPI REST service |
| [ci-bench-L3](https://github.com/ViciusLio/ci-bench-L3) | ~18k lines | Pipeline framework |

Each repo includes ground truth Q&A pairs and evaluation metrics (Hit@K, MRR, MAP, NDCG)
so you can measure exactly how well your pipeline performs.

```bash
# Parse and query ci-bench-L2 in one go
python parse_repo.py ../ci-bench-L2 --output ci_bench_L2_chunks.jsonl
python rag_server.py ci_bench_L2_chunks.jsonl --port 8080

# Then run the benchmark evaluation
cd ../ci-bench-L2
python benchmarks/eval_harness.py --rag-url http://localhost:8080 --output results.json
```

---

## License

MIT — free to use for internal tooling, research, and commercial evaluation.
