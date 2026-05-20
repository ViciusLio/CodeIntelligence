# CodeIntelligence — RAG toolkit for code repositories

> **Query any codebase in natural language — fully local, zero API key, zero data leaving your machine.**

[![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)](#)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

CodeIntelligence is a lightweight RAG (Retrieval-Augmented Generation) toolkit that makes any code repository queryable by an AI model — locally, without sending your code anywhere.

Instead of pasting code into a chat or hoping the model knows your codebase, you:

1. **Parse** the repo into semantic chunks (one per file, function, class)
2. **Embed** the chunks with a local model (optional but recommended)
3. **Query** with a local LLM (Ollama) or Claude API — via CLI or a built-in chat UI

Ask questions like *"Where is JWT authentication implemented?"* or *"Who calls `create_access_token`?"* and get precise, source-cited answers.

---

## v1 vs v2 — What's new

| Feature | v1 | v2 |
|---|---|---|
| Languages parsed | Python only | Python, TypeScript, Go, YAML, Markdown, SQL |
| Context awareness | Code only | Code + environment (deps, git, docker, tests...) |
| Low-confidence detection | None | Automatic — escalates to Claude API with opt-in |
| Test suite | — | 76 tests (unittest, stdlib only) |
| Parser architecture | Single file | Plugin system (`parsers/`) |

---

## Local or Cloud — your choice

| | Local (Ollama) | Cloud (Claude API) |
|--|----------------|--------------------|
| **Script** | `ask_repo_local.py` / `rag_server.py` | `ask_repo.py` |
| **Model** | qwen2.5-coder, codellama, llama3... | claude-opus-4-7 |
| **API key** | not required | `ANTHROPIC_API_KEY` |
| **Data privacy** | stays on your machine | sent to Anthropic |
| **Quality** | good (7B-13B models) | excellent |
| **Cost** | free | pay per token |

---

## Repository structure

```
CodeIntelligence/
|
|-- parse_repo.py            # Python AST parser -> JSONL (original, v1)
|-- parse_repo_multilang.py  # Multi-language parser entry point (v2)
|-- parsers/                 # Language plugins (v2)
|   |-- __init__.py          # Registry + find_parser()
|   |-- python_parser.py     # Wraps parse_repo.py
|   |-- typescript_parser.py # Regex-based
|   |-- go_parser.py         # Regex-based
|   |-- yaml_parser.py       # docker-compose, k8s, GH Actions
|   |-- markdown_parser.py   # H1/H2/H3 sections
|   `-- sql_parser.py        # CREATE statements
|
|-- embed_chunks.py          # Enrich chunks with Ollama embeddings
|-- collect_context.py       # Environment context collector (v2)
|-- escalation.py            # Confidence scoring + Claude escalation (v2)
|
|-- ask_repo.py              # Query via Claude API
|-- ask_repo_local.py        # Query via Ollama (CLI)
|-- rag_server.py            # HTTP server: chat UI + OpenAI/Ollama/Anthropic endpoints
|
|-- rerank.py                # Cross-encoder reranker (optional)
|-- chroma_store.py          # ChromaDB persistent vector store (optional)
|-- run_benchmark.py         # Automated benchmark evaluation
|
`-- tests/                   # 76 tests (unittest, stdlib only)
    |-- test_multilang_parser.py   # 30 tests
    |-- test_context_collector.py  # 25 tests
    `-- test_escalation.py         # 21 tests
```

---

## Quick start (5 minutes)

### Prerequisites

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text   # for semantic retrieval
```

### Step 1 — Parse your repo

**Python only (original parser):**
```bash
python parse_repo.py /path/to/your/repo --output repo_chunks.jsonl
```

**Multi-language (v2):**
```bash
python parse_repo_multilang.py /path/to/your/repo --output repo_chunks.jsonl
# -> summary: Python: 122 files -> 1560 chunks | Markdown: 3 files -> 28 chunks | ...
```

### Step 2 — Embed (recommended)

```bash
python embed_chunks.py repo_chunks.jsonl
# -> writes repo_chunks_embedded.jsonl (~2-15 min depending on repo size)

# After code changes, re-embed only modified files:
python embed_chunks.py repo_chunks.jsonl --incremental --repo-root /path/to/repo
```

### Step 3 — Start the server

```bash
python rag_server.py repo_chunks_embedded.jsonl --embed --rerank --model qwen2.5-coder:7b
```

Open **http://localhost:8080** — the built-in chat UI is ready.

---

## Built-in Chat UI

- Dark theme, streaming tokens visible as they arrive
- Live timer showing response time (also saved in session export)
- Confidence badge on every response
- **Orange escalation banner** when the local model is uncertain (see below)
- Session export to Markdown or JSON (includes response times)
- Enter to send, Shift+Enter for new line

---

## Retrieval modes

### TF-IDF (default, zero setup)
Keyword-based. Fast, no dependencies. Misses semantic matches.

### Semantic (recommended)
Cosine similarity on Ollama embeddings. Finds the right file even when the keywords don't match.

> **Example:** *"Where is JWT token creation implemented?"*
> - TF-IDF: hallucinated `app/utils/jwt.py` (does not exist)
> - Semantic: correctly found `app/core/security.py` with exact claims (`sub`, `iat`, `exp`)

### Semantic + Reranking (best quality)
Over-fetches `top_k * 3` candidates, then re-scores them with a cross-encoder.

```bash
pip install sentence-transformers   # ~80 MB, CPU-friendly, one-time

python rag_server.py repo_chunks_embedded.jsonl --embed --rerank --model qwen2.5-coder:7b
```

---

## v2 Feature: Multi-language parser

Parse repositories that mix Python, TypeScript, Go, YAML, Markdown, and SQL.

```bash
# Parse all supported languages
python parse_repo_multilang.py /my/repo --output chunks.jsonl

# Filter to specific languages
python parse_repo_multilang.py /my/repo --lang python,typescript,go --output chunks.jsonl

# Exclude directories
python parse_repo_multilang.py /my/repo --exclude node_modules --exclude .venv
```

| Language | Detects | Chunk types |
|---|---|---|
| Python | functions, classes, imports, usages | `function`, `class`, `file`, `usages` |
| TypeScript/JS | functions, arrow funcs, classes, interfaces | `function`, `class`, `interface`, `file` |
| Go | funcs (with receivers), structs, interfaces | `function`, `struct`, `interface`, `file` |
| YAML | docker-compose services, k8s manifests, GH Actions jobs | `config_block` |
| Markdown | H1/H2/H3 sections | `doc` |
| SQL | CREATE TABLE/VIEW/FUNCTION/PROCEDURE/INDEX | `schema` |

All parsers are **regex-based, zero external dependencies**, with graceful fallback to a raw file chunk on parse errors.

---

## v2 Feature: Environment context collector

Collect structured context about the runtime environment and merge it with code chunks.
Answers questions like *"Are there missing dependencies?"* or *"What was the last commit about?"*.

```bash
# Collect all context categories
python collect_context.py --repo /my/repo --output ctx.jsonl

# Collect only specific categories
python collect_context.py --repo . --only deps,git,tests --output ctx.jsonl

# Merge with code chunks (context inserted after repo_overview for priority retrieval)
python collect_context.py --repo /my/repo \
    --merge-with repo_chunks.jsonl \
    --output full_chunks.jsonl
```

| Category | What it collects | Chunk type |
|---|---|---|
| `deps` | requirements diff: declared vs installed, MISSING/UNDECLARED | `env_deps` |
| `git` | last 20 commits, active authors, modified files, branch | `git_history`, `git_status` |
| `env` | environment variables (**sensitive values auto-redacted**) | `env_vars` |
| `docker` | running containers vs declared services diff | `docker_state` |
| `process` | relevant running processes (python, uvicorn, redis...) | `process_state` |
| `tests` | last pytest results, coverage %, failed test list | `test_results` |

**Privacy by default:** any env var whose name contains `KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `CREDENTIAL`, `AUTH`, `PRIVATE`, `CERT`, or `SIGNING` is always stored as `=<redacted>`. This cannot be disabled.

---

## v2 Feature: Confidence-based escalation to Claude API

When the local model responds with low confidence, the system detects it automatically and proposes to re-run the query on Claude API — with **explicit user confirmation** before sending any data.

### How it works

1. After each local model response, the server scores confidence in the background
2. Signals of low confidence: uncertainty phrases ("I cannot find", "not present in the context"), no file paths cited, very short answer to a navigation question
3. If score < threshold (default 0.35): orange banner appears under the response
4. User can preview exactly what would be sent (with optional code body anonymization)
5. User clicks "Use Claude API" to confirm — only then is data sent

### Setup

```bash
# Start server with Claude API key
python rag_server.py repo_chunks_embedded.jsonl \
    --embed --rerank \
    --model qwen2.5-coder:7b \
    --claude-api-key sk-ant-...

# Or via environment variable
export ANTHROPIC_API_KEY=sk-ant-...
python rag_server.py repo_chunks_embedded.jsonl --embed --rerank --model qwen2.5-coder:7b
```

The startup log confirms escalation status:
```
Escalation : http://localhost:8080/query/escalate  [enabled, threshold: 0.35]
```

### Escalation API endpoint

```bash
# Preview (no Claude call, no cost)
curl http://localhost:8080/query/escalate \
  -H "Content-Type: application/json" \
  -d '{"question":"...","local_answer":"I cannot find...","chunks_used_ids":[],"confirm":false}'

# Execute (sends to Claude, requires confirm:true)
curl http://localhost:8080/query/escalate \
  -H "Content-Type: application/json" \
  -d '{"question":"...","local_answer":"...","chunks_used_ids":[],"confirm":true,"redact_code":true}'
```

### Escalation threshold

```bash
# Never suggest escalation
python rag_server.py chunks.jsonl --escalation-threshold 0

# Always suggest escalation (useful for testing)
python rag_server.py chunks.jsonl --escalation-threshold 1

# Custom threshold (default: 0.35)
python rag_server.py chunks.jsonl --escalation-threshold 0.5
```

---

## Serving the API

`rag_server.py` exposes four API formats simultaneously:

```
Chat UI           : http://localhost:8080/
OpenAI endpoint   : http://localhost:8080/v1/chat/completions
Anthropic endpoint: http://localhost:8080/v1/messages
Ollama endpoint   : http://localhost:8080/api/chat
Health check      : http://localhost:8080/health
Escalation        : http://localhost:8080/query/escalate
```

| Frontend | Configuration |
|----------|---------------|
| **Browser** | open `http://localhost:8080` |
| **Open WebUI** | Settings -> Connections -> Ollama URL: `http://localhost:8080` |
| **OpenAI SDK** | `base_url="http://localhost:8080/v1"`, `api_key="rag"` |
| **Anthropic SDK** | `base_url="http://localhost:8080"`, `api_key="rag"` |

---

## All options

### parse_repo.py (Python only, original)
```
python parse_repo.py <repo_path> [options]

  --output            Output JSONL file        (default: repo_chunks.jsonl)
  --exclude <pattern> Skip paths matching pattern (repeatable)
  --no-usages         Skip cross-file call-graph chunks
  --include-langs     Also parse: markdown  config  js
```

### parse_repo_multilang.py (multi-language, v2)
```
python parse_repo_multilang.py <repo_path> [options]

  --output            Output JSONL file        (default: repo_chunks.jsonl)
  --lang              Languages to parse       (default: all)
                      Choices: python typescript go yaml markdown sql
  --exclude <pattern> Skip paths matching pattern (repeatable)
```

### embed_chunks.py
```
python embed_chunks.py <chunks.jsonl> [options]

  --output            Output file              (default: <input>_embedded.jsonl)
  --model             Ollama embedding model   (default: nomic-embed-text)
  --url               Ollama base URL          (default: http://localhost:11434)
  --resume            Skip already-embedded chunks
  --incremental       Skip unchanged source files (SHA-256 hashing)
  --repo-root <path>  Source repo root for --incremental file resolution
```

### collect_context.py (v2)
```
python collect_context.py [options]

  --repo <path>       Repository to collect context from  (default: .)
  --output <file>     Output JSONL file
  --only <cats>       Comma-separated: deps,git,env,docker,process,tests
  --merge-with <file> Merge context into existing chunks JSONL
```

### ask_repo_local.py
```
python ask_repo_local.py <chunks.jsonl> ["question"] [options]

  --model             Ollama LLM model         (default: codellama)
  --top-k             Chunks to retrieve       (default: 12)
  --embed             Use semantic retrieval
  --embed-model       Embedding model          (default: nomic-embed-text)
  --rerank            Re-score with cross-encoder (sentence-transformers)
  --no-stream         Print full answer at once
  --all-chunks        Send all chunks as context
```

### rag_server.py
```
python rag_server.py <chunks.jsonl> [options]

  --port              Listening port           (default: 8080)
  --model             Ollama LLM model         (default: qwen2.5-coder:7b)
  --top-k             Chunks per query         (default: 6)
  --embed             Use semantic retrieval
  --embed-model       Embedding model          (default: nomic-embed-text)
  --rerank            Re-score with cross-encoder
  --chroma            Use ChromaDB vector store (requires --embed)
  --claude            Use Claude API as generation backend
  --api-key           Anthropic API key for --claude backend
  --claude-api-key    Anthropic API key for escalation endpoint
  --escalation-threshold  Confidence score threshold (default: 0.35)
                          0 = never escalate, 1 = always escalate
```

---

## Dependencies

All core scripts run with **zero pip installs** (stdlib only: `ast`, `json`, `pathlib`, `urllib`, `http.server`).

Optional packages — install only what you need:

| Package | When needed | Install |
|---|---|---|
| `anthropic` | Claude API backend (`--claude`) or `ask_repo.py` | `pip install anthropic` |
| `sentence-transformers` | Cross-encoder reranking (`--rerank`) | `pip install sentence-transformers` |
| `chromadb` | Persistent vector store (`--chroma`) | `pip install chromadb` |

---

## Benchmark repos

Built alongside three purpose-made benchmark repositories:

| Repo | Size | Domain | Difficulty |
|------|------|--------|------------|
| [ci-bench-L1](https://github.com/ViciusLio/ci-bench-L1) | ~6k lines | Validation library | Easy |
| [ci-bench-L2](https://github.com/ViciusLio/ci-bench-L2) | ~12k lines | FastAPI REST service | Medium |
| [ci-bench-L3](https://github.com/ViciusLio/ci-bench-L3) | ~18k lines | Pipeline framework | Hard |

Each repo has ground truth Q&A pairs and metrics (Hit@K, MRR, MAP, NDCG).

```bash
# Full pipeline on ci-bench-L2
python parse_repo.py ../ci-bench-L2 --output ci_bench_L2_chunks.jsonl
python embed_chunks.py ci_bench_L2_chunks.jsonl
python rag_server.py ci_bench_L2_chunks_embedded.jsonl --embed --rerank --model qwen2.5-coder:7b
# open http://localhost:8080
```

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
# 76 passed
```

---

## License

MIT — free to use for internal tooling, research, and commercial evaluation.
