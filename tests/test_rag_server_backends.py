"""
tests/test_rag_server_backends.py -- Unit tests for the OAI-compatible backend in rag_server.

No real network calls are made (urllib.request.urlopen is mocked).

Run:
    python -m pytest tests/test_rag_server_backends.py -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import rag_server


# ---------------------------------------------------------------------------
# Fake HTTP response helpers
# ---------------------------------------------------------------------------

class _JsonResp:
    """Fake urllib response for non-streaming calls."""
    def __init__(self, obj: dict):
        self._body = json.dumps(obj).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self._body


class _StreamResp:
    """Fake urllib response for streaming calls (yields bytes lines)."""
    def __init__(self, lines: list[str]):
        self._lines = [line.encode() + b"\n" for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(self._lines)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "model": "test-model",
    "ollama_url": "http://ollama:11434",
    "use_oai": False,
    "oai_url": "",
    "oai_embed_url": "",
    "use_claude": False,
    "claude_api_key": "",
    "embed_model": "nomic-embed-text",
    "top_k": 3,
    "use_embed": False,
    "use_rerank": False,
    "use_chroma": False,
    "chroma_collection": None,
    "escalation_threshold": 0.35,
    "escalation_api_key": "",
}


def _cfg(**overrides):
    rag_server.CONFIG.clear()
    rag_server.CONFIG.update({**_BASE_CONFIG, **overrides})
    rag_server.CHUNKS = []


# ---------------------------------------------------------------------------
# 1. _oai_chat_complete
# ---------------------------------------------------------------------------

class TestOAIChatComplete(unittest.TestCase):

    def test_calls_oai_chat_completions_endpoint(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081", model="mistral")
        resp = _JsonResp({"choices": [{"message": {"content": "hello world"}}]})
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            result = rag_server._oai_chat_complete([{"role": "user", "content": "hi"}])

        req = mock_open.call_args[0][0]
        self.assertIn("/v1/chat/completions", req.full_url)
        self.assertEqual(result, "hello world")

    def test_sends_correct_model_and_stream_false(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081", model="llama3")
        resp = _JsonResp({"choices": [{"message": {"content": "ok"}}]})
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            rag_server._oai_chat_complete([{"role": "user", "content": "test"}])

        body = json.loads(mock_open.call_args[0][0].data)
        self.assertEqual(body["model"], "llama3")
        self.assertFalse(body["stream"])

    def test_empty_choices_returns_empty_string(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        resp = _JsonResp({"choices": []})
        with patch("urllib.request.urlopen", return_value=resp):
            result = rag_server._oai_chat_complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# 2. _oai_chat_stream
# ---------------------------------------------------------------------------

class TestOAIChatStream(unittest.TestCase):

    def test_yields_tokens_from_sse(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        sse = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_StreamResp(sse)):
            tokens = list(rag_server._oai_chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(tokens, ["Hello", " world"])

    def test_stops_at_done_sentinel(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        sse = [
            'data: {"choices": [{"delta": {"content": "only"}}]}',
            "data: [DONE]",
            'data: {"choices": [{"delta": {"content": "never"}}]}',
        ]
        with patch("urllib.request.urlopen", return_value=_StreamResp(sse)):
            tokens = list(rag_server._oai_chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual(tokens, ["only"])

    def test_skips_non_data_lines(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        sse = [
            ": keep-alive",
            "",
            'data: {"choices": [{"delta": {"content": "token"}}]}',
            "data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_StreamResp(sse)):
            tokens = list(rag_server._oai_chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual(tokens, ["token"])

    def test_yields_str_not_bytes(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        sse = ['data: {"choices": [{"delta": {"content": "hi"}}]}', "data: [DONE]"]
        with patch("urllib.request.urlopen", return_value=_StreamResp(sse)):
            tokens = list(rag_server._oai_chat_stream([{"role": "user", "content": "x"}]))
        for t in tokens:
            self.assertIsInstance(t, str, "OAI stream must yield str tokens")

    def test_tolerates_malformed_json_lines(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        sse = [
            "data: not-json-at-all",
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: [DONE]",
        ]
        with patch("urllib.request.urlopen", return_value=_StreamResp(sse)):
            tokens = list(rag_server._oai_chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual(tokens, ["ok"])


# ---------------------------------------------------------------------------
# 3. _embed_text — OAI vs Ollama routing
# ---------------------------------------------------------------------------

class TestEmbedText(unittest.TestCase):

    def test_uses_oai_endpoint_when_oai_embed_url_set(self):
        _cfg(oai_embed_url="http://llamacpp:8082")
        embedding = [0.1, 0.2, 0.3]
        with patch("urllib.request.urlopen",
                   return_value=_JsonResp({"data": [{"embedding": embedding}]})) as mock_open:
            result = rag_server._embed_text("hello")

        req = mock_open.call_args[0][0]
        self.assertIn("/v1/embeddings", req.full_url)
        body = json.loads(req.data)
        self.assertEqual(body["input"], "hello")   # OAI uses "input"
        self.assertNotIn("prompt", body)
        self.assertEqual(result, embedding)

    def test_uses_ollama_when_no_oai_embed_url(self):
        _cfg(oai_embed_url="", ollama_url="http://ollama:11434")
        embedding = [0.4, 0.5, 0.6]
        with patch("urllib.request.urlopen",
                   return_value=_JsonResp({"embedding": embedding})) as mock_open:
            result = rag_server._embed_text("hello")

        req = mock_open.call_args[0][0]
        self.assertIn("/api/embeddings", req.full_url)
        body = json.loads(req.data)
        self.assertEqual(body["prompt"], "hello")  # Ollama uses "prompt"
        self.assertNotIn("input", body)
        self.assertEqual(result, embedding)

    def test_oai_embed_uses_embed_model_name(self):
        _cfg(oai_embed_url="http://llamacpp:8082", embed_model="my-embed-model")
        with patch("urllib.request.urlopen",
                   return_value=_JsonResp({"data": [{"embedding": [0.0]}]})) as mock_open:
            rag_server._embed_text("x")

        body = json.loads(mock_open.call_args[0][0].data)
        self.assertEqual(body["model"], "my-embed-model")


# ---------------------------------------------------------------------------
# 4. _chat_complete / _chat_stream routing
# ---------------------------------------------------------------------------

class TestChatRouting(unittest.TestCase):

    def test_chat_complete_routes_to_oai(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        with patch("urllib.request.urlopen",
                   return_value=_JsonResp({"choices": [{"message": {"content": "oai"}}]})) as mock_open:
            result = rag_server._chat_complete([{"role": "user", "content": "q"}])

        req = mock_open.call_args[0][0]
        self.assertIn("/v1/chat/completions", req.full_url)
        self.assertEqual(result, "oai")

    def test_chat_complete_routes_to_ollama_by_default(self):
        _cfg(use_oai=False, use_claude=False, ollama_url="http://ollama:11434")
        with patch("urllib.request.urlopen",
                   return_value=_JsonResp({"message": {"content": "ollama"}})) as mock_open:
            result = rag_server._chat_complete([{"role": "user", "content": "q"}])

        req = mock_open.call_args[0][0]
        self.assertIn("/api/chat", req.full_url)
        self.assertEqual(result, "ollama")

    def test_chat_stream_routes_to_oai_and_yields_str(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081")
        sse = ['data: {"choices": [{"delta": {"content": "hi"}}]}', "data: [DONE]"]
        with patch("urllib.request.urlopen", return_value=_StreamResp(sse)) as mock_open:
            tokens = list(rag_server._chat_stream([{"role": "user", "content": "q"}]))

        req = mock_open.call_args[0][0]
        self.assertIn("/v1/chat/completions", req.full_url)
        self.assertEqual(tokens, ["hi"])
        self.assertIsInstance(tokens[0], str)

    def test_claude_takes_priority_over_oai(self):
        _cfg(use_oai=True, oai_url="http://llamacpp:8081",
             use_claude=True, claude_api_key="sk-test")
        with patch.object(rag_server, "_oai_chat_complete") as mock_oai, \
             patch.object(rag_server, "_claude_chat_complete", return_value="claude"):
            result = rag_server._chat_complete([{"role": "user", "content": "q"}])

        mock_oai.assert_not_called()
        self.assertEqual(result, "claude")

    def test_oai_not_used_when_oai_url_empty(self):
        """use_oai=False (no --oai-url) must not call OAI endpoint."""
        _cfg(use_oai=False, oai_url="", ollama_url="http://ollama:11434")
        with patch("urllib.request.urlopen",
                   return_value=_JsonResp({"message": {"content": "x"}})) as mock_open:
            rag_server._chat_complete([{"role": "user", "content": "q"}])

        req = mock_open.call_args[0][0]
        self.assertNotIn("/v1/chat/completions", req.full_url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
