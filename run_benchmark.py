"""
run_benchmark.py — Automated benchmark evaluation against the RAG server.

Loads ground_truth.json from a ci-bench repo, sends each question to the
running RAG server, checks whether the expected files appear in the retrieved
sources, and prints a detailed report with per-query results and aggregate
retrieval metrics.

Usage:
    python run_benchmark.py --bench <path-to-ci-bench-repo> --server <url>

Options:
    --bench   <path>   Path to ci-bench-L1 / L2 / L3 repo  (required)
    --server  <url>    RAG server base URL  (default: http://localhost:8080)
    --top-k   <n>      Chunks to retrieve   (default: 6)
    --category <cat>   Filter by category   (e.g. direct_retrieval, cross_layer)
    --output  <file>   Save JSON report to file

Metrics computed:
    Hit@K   — at least one expected file in top-K retrieved chunks
    MRR     — mean reciprocal rank of first relevant chunk
    MAP     — mean average precision
    NDCG@K  — normalised discounted cumulative gain

Requires: RAG server running (python rag_server.py ...)
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Metrics (identical logic to benchmarks/metrics.py in ci-bench repos)
# ---------------------------------------------------------------------------

def _file_from_source(source: dict) -> str:
    """Extract file path from a chunk source dict returned by /query."""
    # source["id"] is like "function::app/core/security.py::create_access_token"
    # or "file::app/core/security.py" or "class::..."
    chunk_id = source.get("id", "")
    parts = chunk_id.split("::")
    if len(parts) >= 2:
        return parts[1]   # always the file path
    return chunk_id


def hit_at_k(retrieved_files: list[str], expected_files: list[str]) -> float:
    expected = set(expected_files)
    return 1.0 if any(f in expected for f in retrieved_files) else 0.0


def reciprocal_rank(retrieved_files: list[str], expected_files: list[str]) -> float:
    expected = set(expected_files)
    for i, f in enumerate(retrieved_files, 1):
        if f in expected:
            return 1.0 / i
    return 0.0


def average_precision(retrieved_files: list[str], expected_files: list[str]) -> float:
    expected = set(expected_files)
    hits = 0
    precision_sum = 0.0
    for i, f in enumerate(retrieved_files, 1):
        if f in expected:
            hits += 1
            precision_sum += hits / i
    return precision_sum / max(len(expected), 1)


def ndcg_at_k(retrieved_files: list[str], expected_files: list[str]) -> float:
    expected = set(expected_files)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, f in enumerate(retrieved_files)
        if f in expected
    )
    ideal_dcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(expected), len(retrieved_files)))
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Server query
# ---------------------------------------------------------------------------

def query_server(question: str, base_url: str, top_k: int) -> tuple[str, list[dict]]:
    """Call /query and return (answer, sources)."""
    payload = json.dumps({"question": question, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return result.get("answer", ""), result.get("sources", [])
    except urllib.error.URLError as exc:
        print(f"  [error] cannot reach server: {exc}", file=sys.stderr)
        return "", []


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"

def render_report(results: list[dict], category_filter: str | None) -> None:
    total = len(results)
    if total == 0:
        print("No queries to evaluate.")
        return

    hits    = [r["hit_at_k"]  for r in results]
    mrrs    = [r["mrr"]       for r in results]
    maps    = [r["ap"]        for r in results]
    ndcgs   = [r["ndcg"]      for r in results]

    avg_hit  = sum(hits)  / total
    avg_mrr  = sum(mrrs)  / total
    avg_map  = sum(maps)  / total
    avg_ndcg = sum(ndcgs) / total

    print()
    print("=" * 70)
    print(" BENCHMARK RESULTS")
    if category_filter:
        print(f" Category filter: {category_filter}")
    print("=" * 70)
    print()

    # per-query table
    print(f"  {'ID':<6} {'Diff':<8} {'Category':<22} {'Hit':<5} {'MRR':<6} {'NDCG':<6}  Question")
    print(f"  {'-'*6} {'-'*8} {'-'*22} {'-'*5} {'-'*6} {'-'*6}  {'-'*30}")

    for r in results:
        icon  = PASS if r["hit_at_k"] == 1.0 else FAIL
        q_short = r["question"][:45] + ("…" if len(r["question"]) > 45 else "")
        print(
            f"  {r['id']:<6} {r['difficulty']:<8} {r['category']:<22} "
            f"{icon:<5} {r['mrr']:<6.2f} {r['ndcg']:<6.2f}  {q_short}"
        )
        # show expected vs found files
        expected = set(r["expected_files"])
        found    = set(r["retrieved_files"])
        correct  = expected & found
        missing  = expected - found
        if correct:
            for f in sorted(correct):
                print(f"           + {f}")
        if missing:
            for f in sorted(missing):
                print(f"           - {f}  (not retrieved)")

    print()
    print("=" * 70)
    print(" AGGREGATE METRICS")
    print("=" * 70)
    print(f"  Queries evaluated : {total}")
    print(f"  Hit@K             : {avg_hit:.3f}   (target >= 0.75)")
    print(f"  MRR               : {avg_mrr:.3f}   (target >= 0.60)")
    print(f"  MAP               : {avg_map:.3f}   (target >= 0.55)")
    print(f"  NDCG@K            : {avg_ndcg:.3f}   (target >= 0.58)")
    print()

    # by category
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    if len(categories) > 1:
        print("  By category:")
        for cat, items in sorted(categories.items()):
            cat_hit = sum(i["hit_at_k"] for i in items) / len(items)
            cat_mrr = sum(i["mrr"]      for i in items) / len(items)
            icon = PASS if cat_hit >= 0.75 else (WARN if cat_hit >= 0.5 else FAIL)
            print(f"    {icon} {cat:<25} Hit@K={cat_hit:.2f}  MRR={cat_mrr:.2f}  ({len(items)} queries)")

    # by difficulty
    difficulties: dict[str, list[dict]] = {}
    for r in results:
        difficulties.setdefault(r.get("difficulty", "?"), []).append(r)

    if len(difficulties) > 1:
        print()
        print("  By difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if diff not in difficulties:
                continue
            items = difficulties[diff]
            d_hit = sum(i["hit_at_k"] for i in items) / len(items)
            d_mrr = sum(i["mrr"]      for i in items) / len(items)
            print(f"    {diff:<10} Hit@K={d_hit:.2f}  MRR={d_mrr:.2f}  ({len(items)} queries)")

    print()
    overall = PASS if avg_hit >= 0.75 and avg_mrr >= 0.60 else FAIL
    print(f"  Overall: {overall}")
    print("=" * 70)
    print()


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

    bench_path  = _pop_arg(args, "--bench")
    server_url  = _pop_arg(args, "--server")   or "http://localhost:8080"
    top_k       = int(_pop_arg(args, "--top-k") or 6)
    category    = _pop_arg(args, "--category")
    output_file = _pop_arg(args, "--output")

    if not bench_path:
        print("Error: --bench <path-to-ci-bench-repo> is required", file=sys.stderr)
        sys.exit(1)

    ground_truth_path = Path(bench_path) / "benchmarks" / "ground_truth.json"
    if not ground_truth_path.exists():
        print(f"Error: ground_truth.json not found at {ground_truth_path}", file=sys.stderr)
        sys.exit(1)

    # verify server is up
    try:
        with urllib.request.urlopen(f"{server_url}/health", timeout=5) as r:
            info = json.loads(r.read())
        print(f"Server  : {server_url}")
        print(f"Model   : {info.get('model')}  ({info.get('backend')})  "
              f"retrieval={info.get('retrieval')}  chunks={info.get('chunks')}")
    except Exception as exc:
        print(f"Error: cannot reach RAG server at {server_url}: {exc}", file=sys.stderr)
        print("Start it first:  python rag_server.py <chunks.jsonl>", file=sys.stderr)
        sys.exit(1)

    # load ground truth
    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    queries = gt.get("queries", [])
    if category:
        queries = [q for q in queries if q.get("category") == category]

    print(f"Bench   : {Path(bench_path).name}  ({len(queries)} queries)")
    print(f"top_k   : {top_k}")
    print()

    results: list[dict] = []
    t0 = time.time()

    for i, q in enumerate(queries, 1):
        qid      = q.get("id", f"q{i:02d}")
        question = q.get("question", "")
        expected = q.get("expected_files", [])
        diff     = q.get("difficulty", "?")
        cat      = q.get("category", "?")

        print(f"  [{i:02d}/{len(queries)}] {qid} — {question[:60]}{'…' if len(question)>60 else ''}")

        answer, sources = query_server(question, server_url, top_k)
        retrieved_files = [_file_from_source(s) for s in sources]

        hit  = hit_at_k(retrieved_files, expected)
        mrr  = reciprocal_rank(retrieved_files, expected)
        ap   = average_precision(retrieved_files, expected)
        ndcg = ndcg_at_k(retrieved_files, expected)

        icon = PASS if hit == 1.0 else FAIL
        print(f"         {icon}  Hit@K={hit:.0f}  MRR={mrr:.2f}  NDCG={ndcg:.2f}")

        results.append({
            "id": qid,
            "question": question,
            "category": cat,
            "difficulty": diff,
            "expected_files": expected,
            "retrieved_files": retrieved_files,
            "answer": answer,
            "hit_at_k": hit,
            "mrr": mrr,
            "ap": ap,
            "ndcg": ndcg,
        })

    elapsed = time.time() - t0
    print(f"\n  Completed {len(results)} queries in {elapsed:.1f}s")

    render_report(results, category)

    if output_file:
        out = {
            "server": server_url,
            "bench": Path(bench_path).name,
            "top_k": top_k,
            "category_filter": category,
            "elapsed_seconds": round(elapsed, 1),
            "metrics": {
                "hit_at_k": sum(r["hit_at_k"] for r in results) / len(results),
                "mrr":      sum(r["mrr"]      for r in results) / len(results),
                "map":      sum(r["ap"]        for r in results) / len(results),
                "ndcg":     sum(r["ndcg"]      for r in results) / len(results),
            },
            "queries": results,
        }
        Path(output_file).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Report saved to {output_file}")


if __name__ == "__main__":
    main()
