"""
tests/test_escalation.py -- Unit tests for escalation.py

All fixtures are inline strings.  No network calls are made.

Run:
    python -m pytest tests/test_escalation.py -v
    # or
    python -m unittest tests.test_escalation -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from escalation import (
    score_confidence,
    build_escalation_payload,
    _redact_body,
    _maybe_redact_chunk,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str = "function::app/core/auth.py::get_token",
    chunk_type: str = "function",
    text: str = "def get_token(user_id):\n    return db.query(user_id)\n",
) -> dict:
    return {"id": chunk_id, "type": chunk_type, "text": text}


FUNCTION_CHUNK_WITH_BODY = _make_chunk(
    text=(
        "def create_access_token(data: dict, expires_delta=None):\n"
        '    """Create a signed JWT token."""\n'
        "    to_encode = data.copy()\n"
        "    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))\n"
        "    to_encode.update({'exp': expire})\n"
        "    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n"
    )
)

THREE_CHUNKS = [
    _make_chunk("repo_overview", "repo_overview", "Overview text"),
    _make_chunk("function::app/main.py::main", "function", "def main(): pass"),
    _make_chunk("file::app/auth.py", "file", "# auth module"),
]


# ---------------------------------------------------------------------------
# 1. test_low_confidence_cannot_find
# ---------------------------------------------------------------------------

class TestLowConfidenceCannotFind(unittest.TestCase):

    def test_low_confidence_cannot_find(self):
        answer = "I cannot find the implementation of this function in the provided context."
        result = score_confidence(answer, "Where is the JWT function?", THREE_CHUNKS)
        self.assertLess(result["score"], 0.35, f"Expected low score, got {result['score']}")
        self.assertTrue(result["should_escalate"])
        self.assertTrue(
            any("cannot find" in s.lower() for s in result["signals"]),
            f"Expected 'cannot find' signal, got {result['signals']}"
        )


# ---------------------------------------------------------------------------
# 2. test_low_confidence_short_answer
# ---------------------------------------------------------------------------

class TestLowConfidenceShortAnswer(unittest.TestCase):

    def test_low_confidence_short_answer(self):
        # < 80 words, question contains "where"
        answer = "The function might be somewhere in the codebase."
        question = "Where is the rate limiter implemented?"
        result = score_confidence(answer, question, [THREE_CHUNKS[0]])
        self.assertLess(result["score"], 0.65)
        self.assertTrue(
            any("short" in s.lower() or "navigation" in s.lower() for s in result["signals"]),
            f"Expected structural signal, got {result['signals']}"
        )


# ---------------------------------------------------------------------------
# 3. test_high_confidence_with_files
# ---------------------------------------------------------------------------

class TestHighConfidenceWithFiles(unittest.TestCase):

    def test_high_confidence_with_files(self):
        answer = (
            "The JWT implementation lives in `app/core/security.py` inside the "
            "`create_access_token` function. The token is verified in "
            "app/dependencies/auth.py using `verify_token`. Both files use the "
            "HS256 algorithm and share the SECRET_KEY from settings."
        )
        result = score_confidence(answer, "Where is JWT implemented?", THREE_CHUNKS)
        self.assertGreater(result["score"], 0.5)
        self.assertFalse(result["should_escalate"])


# ---------------------------------------------------------------------------
# 4. test_high_confidence_with_code_block
# ---------------------------------------------------------------------------

class TestHighConfidenceWithCodeBlock(unittest.TestCase):

    def test_high_confidence_with_code_block(self):
        answer = (
            "The token creation is in app/core/security.py:\n\n"
            "```python\n"
            "def create_access_token(data: dict) -> str:\n"
            "    return jwt.encode(data, SECRET_KEY)\n"
            "```\n"
            "This is called from app/routers/auth.py."
        )
        result = score_confidence(answer, "How is the token created?", THREE_CHUNKS)
        # code fence should contribute positively
        self.assertTrue(
            any("code snippet" in s.lower() for s in result["signals"]),
            f"Expected code snippet signal, got {result['signals']}"
        )
        self.assertGreater(result["score"], 0.35)


# ---------------------------------------------------------------------------
# 5. test_no_signals_neutral
# ---------------------------------------------------------------------------

class TestNoSignalsNeutral(unittest.TestCase):

    def test_no_signals_neutral(self):
        # Generic response with no strong positive or negative markers
        answer = (
            "The repository contains a FastAPI application with several modules. "
            "The main entry point sets up the application and registers routers."
        )
        result = score_confidence(answer, "Tell me about the project structure.", THREE_CHUNKS)
        # Should not escalate -- score neither hits certainty floor nor ceiling hard
        self.assertFalse(result["should_escalate"])


# ---------------------------------------------------------------------------
# 6. test_redact_code_hides_body
# ---------------------------------------------------------------------------

class TestRedactCodeHidesBody(unittest.TestCase):

    def test_redact_code_hides_body(self):
        chunk = _make_chunk(text=FUNCTION_CHUNK_WITH_BODY["text"])
        result = _maybe_redact_chunk(chunk, redact=True)
        self.assertIn("[REDACTED - function body hidden]", result["text"])
        self.assertNotIn("to_encode = data.copy()", result["text"])
        self.assertNotIn("jwt.encode", result["text"])


# ---------------------------------------------------------------------------
# 7. test_redact_code_keeps_signature
# ---------------------------------------------------------------------------

class TestRedactCodeKeepsSignature(unittest.TestCase):

    def test_redact_code_keeps_signature(self):
        chunk = _make_chunk(text=FUNCTION_CHUNK_WITH_BODY["text"])
        result = _maybe_redact_chunk(chunk, redact=True)
        # The def line (signature) must still be present
        self.assertIn("def create_access_token", result["text"])

    def test_redact_code_keeps_docstring(self):
        chunk = _make_chunk(text=FUNCTION_CHUNK_WITH_BODY["text"])
        result = _maybe_redact_chunk(chunk, redact=True)
        # The docstring must still be present
        self.assertIn("Create a signed JWT token", result["text"])


# ---------------------------------------------------------------------------
# 8. test_no_redact_keeps_body
# ---------------------------------------------------------------------------

class TestNoRedactKeepsBody(unittest.TestCase):

    def test_no_redact_keeps_body(self):
        chunk = _make_chunk(text=FUNCTION_CHUNK_WITH_BODY["text"])
        result = _maybe_redact_chunk(chunk, redact=False)
        self.assertEqual(result["text"], FUNCTION_CHUNK_WITH_BODY["text"])
        self.assertNotIn("[REDACTED", result["text"])

    def test_no_redact_file_chunk_unchanged(self):
        file_chunk = _make_chunk(
            chunk_id="file::app/main.py",
            chunk_type="file",
            text="# main module\nimport fastapi\n",
        )
        result = _maybe_redact_chunk(file_chunk, redact=True)
        # file type is not redacted at all
        self.assertEqual(result["text"], file_chunk["text"])


# ---------------------------------------------------------------------------
# 9. test_build_payload_model_name
# ---------------------------------------------------------------------------

class TestBuildPayloadModelName(unittest.TestCase):

    def test_build_payload_model_name(self):
        result = build_escalation_payload(
            question="Where is auth?",
            chunks=THREE_CHUNKS,
            local_answer="I cannot find it.",
        )
        self.assertEqual(result["payload"]["model"], "claude-opus-4-7")


# ---------------------------------------------------------------------------
# 10. test_build_payload_thinking_adaptive
# ---------------------------------------------------------------------------

class TestBuildPayloadThinkingAdaptive(unittest.TestCase):

    def test_build_payload_thinking_adaptive(self):
        result = build_escalation_payload(
            question="Where is auth?",
            chunks=THREE_CHUNKS,
            local_answer="I cannot find it.",
        )
        thinking = result["payload"].get("thinking")
        self.assertIsNotNone(thinking)
        self.assertEqual(thinking.get("type"), "adaptive")


# ---------------------------------------------------------------------------
# 11. test_estimated_tokens_calculation
# ---------------------------------------------------------------------------

class TestEstimatedTokensCalculation(unittest.TestCase):

    def test_estimated_tokens_calculation(self):
        result = build_escalation_payload(
            question="Q",
            chunks=[THREE_CHUNKS[0]],
            local_answer="A",
        )
        # estimated_tokens must be len(system+user_text) // 4
        # We can't know the exact value without recomputing, but it must be > 0
        self.assertGreater(result["estimated_tokens"], 0)
        # And must be an int
        self.assertIsInstance(result["estimated_tokens"], int)

    def test_estimated_tokens_grows_with_more_chunks(self):
        small = build_escalation_payload("Q", [THREE_CHUNKS[0]], "A")
        large = build_escalation_payload("Q", THREE_CHUNKS, "A")
        self.assertGreater(large["estimated_tokens"], small["estimated_tokens"])


# ---------------------------------------------------------------------------
# 12. test_score_clamp_between_0_and_1
# ---------------------------------------------------------------------------

class TestScoreClamp(unittest.TestCase):

    def test_score_cannot_exceed_1(self):
        # Very confident answer -- many positive signals
        answer = (
            "The JWT logic is in `app/core/security.py`. See `create_access_token`.\n"
            "The verification is in `app/deps/auth.py` via `verify_token`.\n"
            "```python\n"
            "def create_access_token(data):\n"
            "    return jwt.encode(data, SECRET_KEY)\n"
            "```\n"
            "Both modules import from app/config/settings.py which holds SECRET_KEY. "
            "The token expiry is configured in app/core/security.py as well. "
            "Overall the flow is: request -> app/routers/auth.py -> "
            "app/core/security.py -> app/models/user.py.\n"
        )
        result = score_confidence(answer, "Where is JWT implemented?", THREE_CHUNKS)
        self.assertLessEqual(result["score"], 1.0)
        self.assertGreaterEqual(result["score"], 0.0)

    def test_score_cannot_go_below_0(self):
        # Worst case: many strong uncertainty phrases
        answer = (
            "I cannot find this. I cannot determine the answer. "
            "I'm not sure and it's unclear. I apologize, i cannot answer. "
            "I'm unable to locate it. Not mentioned anywhere. Not specified."
        )
        result = score_confidence(answer, "Where is the function?", [])
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertTrue(result["should_escalate"])

    def test_score_is_float(self):
        answer = "The function is defined in app/utils.py."
        result = score_confidence(answer, "What does utils do?", THREE_CHUNKS)
        self.assertIsInstance(result["score"], float)


# ---------------------------------------------------------------------------
# Extra: preview contains expected sections
# ---------------------------------------------------------------------------

class TestPreviewContent(unittest.TestCase):

    def test_preview_contains_question(self):
        result = build_escalation_payload(
            question="Where is the login endpoint?",
            chunks=THREE_CHUNKS,
            local_answer="I cannot find it.",
        )
        self.assertIn("Where is the login endpoint?", result["preview"])

    def test_preview_contains_chunk_ids(self):
        result = build_escalation_payload(
            question="Q",
            chunks=THREE_CHUNKS,
            local_answer="A",
        )
        for chunk in THREE_CHUNKS:
            self.assertIn(chunk["id"], result["preview"])

    def test_preview_redact_label(self):
        on = build_escalation_payload("Q", THREE_CHUNKS, "A", redact_code=True)
        off = build_escalation_payload("Q", THREE_CHUNKS, "A", redact_code=False)
        self.assertIn("ON", on["preview"])
        self.assertIn("OFF", off["preview"])

    def test_redacted_flag_in_result(self):
        on = build_escalation_payload("Q", THREE_CHUNKS, "A", redact_code=True)
        off = build_escalation_payload("Q", THREE_CHUNKS, "A", redact_code=False)
        self.assertTrue(on["redacted"])
        self.assertFalse(off["redacted"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
