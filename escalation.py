"""
escalation.py -- Confidence-based escalation for RAG responses.

When the local model replies with low confidence, these utilities detect
the signals, build a preview of what would be sent to Claude API, and
(with explicit user confirmation) execute the escalation call.

Zero external dependencies -- urllib, json, re are all stdlib.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESCALATION_THRESHOLD = 0.35   # score below this => should_escalate = True

# Phrases that signal the model is uncertain (case-insensitive)
_UNCERTAINTY_PHRASES_STRONG = [
    "i cannot find",
    "i don't see",
    "not found in the context",
    "not present in",
    "i'm not sure",
    "i cannot determine",
    "i'm unable to",
    "i don't have enough",
    "it's unclear",
    "cannot be determined",
    "not mentioned",
    "not specified",
    "i apologize",
    "i cannot answer",
    "outside the scope",
    "based on limited",
    "partial information",
]

_UNCERTAINTY_PHRASES_WEAK = [
    "might be",
    "possibly",
    "perhaps",
]

# Navigation / structural questions that expect a concrete file path answer
_NAVIGATION_KEYWORDS = ["where", "how", "which", "what function", "trace"]

# Pattern for file paths in responses (e.g., app/core/security.py)
_FILE_PATH_RE = re.compile(r"\b\w[\w/.-]*\w\.py\b")

# Pattern for backtick-quoted identifiers
_BACKTICK_RE = re.compile(r"`[A-Za-z_]\w*`")

# Pattern for code fences
_CODE_FENCE_RE = re.compile(r"```")

SYSTEM_PROMPT = (
    "You are an expert Python software engineer and code analyst. "
    "Answer questions about the Python repository using ONLY the context chunks provided. "
    "Each chunk is labelled with its type (REPO_OVERVIEW, FILE, FUNCTION, CLASS) and id. "
    "Be precise, cite file/function/class names when relevant. "
    "If the answer is not in the context, say so explicitly."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_confidence(
    answer: str,
    question: str,
    chunks_used: list[dict],
) -> dict:
    """
    Score the confidence of a local-model answer.

    Returns:
        {
            "score": float,          # 0.0 = no confidence, 1.0 = high confidence
            "signals": list[str],    # human-readable signals found
            "should_escalate": bool, # True if score < ESCALATION_THRESHOLD
            "reason": str            # short explanation
        }
    """
    score = 1.0
    signals: list[str] = []
    answer_lower = answer.lower()
    question_lower = question.lower()

    # ---- Negative signals: explicit uncertainty phrases (strong) ----
    strong_hits = 0
    for phrase in _UNCERTAINTY_PHRASES_STRONG:
        if phrase in answer_lower:
            if strong_hits < 2:   # cap contribution at -0.70 total
                score -= 0.35
                strong_hits += 1
            signals.append(f"Expressed uncertainty: '{phrase}'")

    # ---- Negative signals: weak hedging phrases ----
    for phrase in _UNCERTAINTY_PHRASES_WEAK:
        if phrase in answer_lower:
            score -= 0.175   # 0.5x weight vs strong = 0.35 * 0.5
            signals.append(f"Hedging language: '{phrase}'")

    # ---- Negative structural signals ----
    word_count = len(answer.split())

    # Short answer to a navigation question
    has_nav_keyword = any(kw in question_lower for kw in _NAVIGATION_KEYWORDS)
    if word_count < 80 and has_nav_keyword:
        score -= 0.25
        signals.append("Short answer (<80 words) to a navigation/tracing question")

    # No file paths cited when question implies code location
    file_paths_in_answer = _FILE_PATH_RE.findall(answer)
    if len(file_paths_in_answer) == 0 and has_nav_keyword:
        score -= 0.15
        signals.append("No file paths cited in answer")

    # Few chunks used
    if len(chunks_used) < 3:
        score -= 0.1
        signals.append(f"Fewer than 3 chunks used ({len(chunks_used)} used)")

    # ---- Positive signals ----
    # 2+ specific file paths
    unique_paths = set(_FILE_PATH_RE.findall(answer))
    if len(unique_paths) >= 2:
        score += 0.2
        signals.append(f"Cites {len(unique_paths)} specific file paths")
    elif len(unique_paths) == 1:
        score += 0.1
        signals.append(f"Cites 1 specific file path")

    # At least 1 backtick-quoted identifier
    backtick_matches = _BACKTICK_RE.findall(answer)
    if backtick_matches:
        score += 0.15
        signals.append(f"References named identifiers (e.g., {backtick_matches[0]})")

    # Long answer
    if word_count > 200:
        score += 0.15
        signals.append(f"Detailed answer ({word_count} words)")

    # Code snippet present
    if _CODE_FENCE_RE.search(answer):
        score += 0.1
        signals.append("Contains code snippet (```)")

    # Clamp
    score = max(0.0, min(1.0, score))

    should_escalate = score < ESCALATION_THRESHOLD

    if not signals:
        reason = "No strong confidence signals detected"
    elif should_escalate:
        reason = "; ".join(signals[:2])
    else:
        reason = "Local model appears confident"

    return {
        "score": round(score, 4),
        "signals": signals,
        "should_escalate": should_escalate,
        "reason": reason,
    }


def build_escalation_payload(
    question: str,
    chunks: list[dict],
    local_answer: str,
    redact_code: bool = True,
) -> dict:
    """
    Build a preview and Anthropic API payload for escalation.

    Returns:
        {
            "preview": str,           # human-readable summary of what would be sent
            "payload": dict,          # Anthropic Messages API payload
            "estimated_tokens": int,  # rough estimate: len(text) // 4
            "redacted": bool
        }
    """
    processed_chunks = [_maybe_redact_chunk(c, redact_code) for c in chunks]

    context_parts = []
    for c in processed_chunks:
        header = f"[{c['type'].upper()}] {c['id']}"
        context_parts.append(f"{header}\n{c['text']}")
    context_text = "\n\n---\n\n".join(context_parts)

    user_message_text = f"Context:\n\n{context_text}\n\n---\n\nQuestion: {question}"

    full_text_for_tokens = SYSTEM_PROMPT + user_message_text
    estimated_tokens = len(full_text_for_tokens) // 4

    cost_usd = estimated_tokens * 0.000005

    # Build chunk listing lines for preview
    chunk_lines = []
    for c in chunks:
        chunk_type = c.get("type", "unknown").upper()
        chunk_id = c.get("id", "?")
        chunk_lines.append(f"  [{chunk_type}] {chunk_id}")
    chunk_listing = "\n".join(chunk_lines)

    redact_label = "ON (function bodies hidden)" if redact_code else "OFF"

    preview = (
        "=== ESCALATION PREVIEW ===\n"
        "This query will be sent to Claude API (claude-opus-4-7).\n"
        "\n"
        f'Question: "{question}"\n'
        "\n"
        f"Context chunks to be sent ({len(chunks)} chunks, ~{estimated_tokens} tokens):\n"
        f"{chunk_listing}\n"
        "\n"
        f"Estimated cost: ~${cost_usd:.4f} (input) + response\n"
        f"Code redaction: {redact_label}\n"
        "==========================="
    )

    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_message_text}
        ],
    }

    return {
        "preview": preview,
        "payload": payload,
        "estimated_tokens": estimated_tokens,
        "redacted": redact_code,
    }


def escalate_to_claude(
    payload: dict,
    api_key: str,
    stream: bool = False,
) -> str:
    """
    Call Claude API (claude-opus-4-7) with the given payload using urllib.

    Raises ValueError on 4xx/5xx with a descriptive message.
    Raises ConnectionError if the server is unreachable.
    """
    url = "https://api.anthropic.com/v1/messages"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "interleaved-thinking-2025-05-14",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
            err_msg = err_body.get("error", {}).get("message", str(exc))
        except Exception:
            err_msg = str(exc)
        if status == 401:
            raise ValueError(f"Claude API: authentication failed (401) -- check your API key") from exc
        elif status == 429:
            raise ValueError(f"Claude API: rate limited (429) -- retry later") from exc
        elif status >= 500:
            raise ValueError(f"Claude API: server error ({status}): {err_msg}") from exc
        else:
            raise ValueError(f"Claude API: HTTP {status}: {err_msg}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Claude API: cannot connect to {url}: {exc}") from exc

    # Extract text from content blocks (skip thinking blocks)
    parts = []
    for block in body.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_redact_chunk(chunk: dict, redact: bool) -> dict:
    """
    Return a (shallow) copy of the chunk with the function/class body
    replaced by a placeholder when redact=True.
    """
    if not redact:
        return chunk
    if chunk.get("type") not in ("function", "class"):
        return chunk

    text = chunk.get("text", "")
    redacted_text = _redact_body(text)
    if redacted_text is text:
        return chunk   # nothing to redact

    result = dict(chunk)
    result["text"] = redacted_text
    return result


def _redact_body(text: str) -> str:
    """
    Replace the body of a function or class definition with [REDACTED].

    Keeps:  the first `def `/ `class ` line (signature), any immediately
            following docstring (first triple-quoted block), and decorators
            before the def/class line.
    Replaces: everything after the signature (and optional docstring).
    """
    lines = text.splitlines(keepends=True)

    # Find the line that starts the def/class
    def_index = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            def_index = i
            break

    if def_index is None:
        return text   # no def/class found

    # Keep everything up to and including the def/class line
    kept = lines[: def_index + 1]

    # Check if the very next non-empty line is a docstring
    rest = lines[def_index + 1:]
    docstring_end = _find_docstring_end(rest)
    if docstring_end is not None:
        kept.extend(rest[: docstring_end + 1])

    kept.append("    [REDACTED - function body hidden]\n")
    return "".join(kept)


def _find_docstring_end(lines: list[str]) -> int | None:
    """
    If `lines` starts with an (indented) triple-quoted docstring, return the
    index of the last line of that docstring.  Otherwise return None.
    """
    # skip leading blank lines
    start = 0
    while start < len(lines) and lines[start].strip() == "":
        start += 1

    if start >= len(lines):
        return None

    first = lines[start].lstrip()
    # Only recognise triple-double-quote docstrings (standard)
    for q in ('"""', "'''"):
        if first.startswith(q):
            opener = q
            break
    else:
        return None

    # Check if the whole docstring fits on one line
    tail = first[len(opener):]
    if opener in tail:
        return start   # single-line docstring

    # Multi-line: scan for the closing triple quote
    for i in range(start + 1, len(lines)):
        if opener in lines[i]:
            return i

    return None   # unclosed docstring -- don't try to keep it
