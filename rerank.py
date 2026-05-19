"""
rerank.py — Cross-encoder reranker for CodeIntelligence RAG pipeline.

Uses a local cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2, ~80 MB,
CPU-friendly) from the sentence-transformers library to re-score retrieved chunks.

Typical usage:
    from rerank import rerank
    reranked = rerank(query, chunks, top_n=5)

The reranker is opt-in: if sentence-transformers is not installed, a warning is
logged and the original chunk list is returned unchanged so the system degrades
gracefully without crashing.
"""

from __future__ import annotations

import logging
import warnings
from typing import List

logger = logging.getLogger(__name__)

# Default cross-encoder model (~80 MB, CPU-friendly)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level cache so the model is only loaded once per process
_cached_model = None
_cached_model_name: str | None = None


def _load_model(model_name: str):
    """Load and cache the cross-encoder model.  Returns None if unavailable."""
    global _cached_model, _cached_model_name

    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model

    try:
        # Suppress noisy FutureWarnings from transformers / tokenizers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import CrossEncoder  # type: ignore

        logger.info("Loading cross-encoder model: %s", model_name)
        _cached_model = CrossEncoder(model_name)
        _cached_model_name = model_name
        return _cached_model
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed. "
            "Reranking is disabled — install it with: pip install sentence-transformers\n"
            "Falling back to the original chunk order."
        )
        return None
    except Exception as exc:
        logger.warning(
            "Failed to load cross-encoder model %r: %s. "
            "Falling back to original chunk order.",
            model_name,
            exc,
        )
        return None


def rerank(
    query: str,
    chunks: List[dict],
    top_n: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> List[dict]:
    """
    Re-rank a list of retrieved chunks using a cross-encoder.

    Parameters
    ----------
    query:
        The user's natural-language question.
    chunks:
        List of chunk dicts.  Each dict must have at least ``"text"`` and ``"id"`` fields.
    top_n:
        Number of chunks to return after reranking.
    model_name:
        HuggingFace model ID for the CrossEncoder (default: ms-marco-MiniLM-L-6-v2).

    Returns
    -------
    List of up to ``top_n`` chunk dicts, sorted by cross-encoder score (highest first).
    Each returned dict has an extra ``"score"`` field with the float cross-encoder score.

    If sentence-transformers is not available, returns the first ``top_n`` chunks from
    the original list unchanged (with a ``"score": None`` field appended).
    """
    if not chunks:
        return []

    model = _load_model(model_name)

    if model is None:
        # Graceful degradation: return top_n chunks without scoring
        result = []
        for chunk in chunks[:top_n]:
            out = dict(chunk)
            out["score"] = None
            result.append(out)
        return result

    # Build (query, text) pairs for scoring
    pairs = [(query, chunk["text"]) for chunk in chunks]

    try:
        scores = model.predict(pairs)
    except Exception as exc:
        logger.warning(
            "Cross-encoder scoring failed: %s. Falling back to original order.", exc
        )
        result = []
        for chunk in chunks[:top_n]:
            out = dict(chunk)
            out["score"] = None
            result.append(out)
        return result

    # Attach scores and sort descending
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        out = dict(chunk)
        out["score"] = float(score)
        scored_chunks.append(out)

    scored_chunks.sort(key=lambda c: c["score"], reverse=True)
    return scored_chunks[:top_n]
