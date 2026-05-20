"""
tests/test_context_collector.py -- Unit tests for collect_context.py

All fixtures are inline strings (no real files on disk except via tempfile).
No real subprocess calls -- subprocess.run is mocked where needed.

Run:
    python -m pytest tests/test_context_collector.py -v
    # or
    python -m unittest tests.test_context_collector -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collect_context import (
    _parse_requirements_txt,
    _redact_value,
    _parse_compose_services,
    _parse_junit_xml,
    _parse_coverage_xml,
    collect_all,
    deps_context,
    git_context,
    env_context,
    docker_context,
    process_context,
    testresults_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(files: dict[str, str]) -> Path:
    """Create a temp directory with the given {filename: content} map."""
    tmp = tempfile.mkdtemp()
    root = Path(tmp)
    for fname, content in files.items():
        fpath = root / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
    return root


def _fake_run_ok(output: str):
    """Return a side_effect callable that returns (True, output) always."""
    def _inner(cmd, timeout=10, cwd=None):
        return True, output
    return _inner


def _fake_run_fail():
    """Return a side_effect callable that returns (False, '') always."""
    def _inner(cmd, timeout=10, cwd=None):
        return False, ""
    return _inner


# ---------------------------------------------------------------------------
# 1. test_deps_requirements_parsing
# ---------------------------------------------------------------------------

class TestDepsRequirementsParsing(unittest.TestCase):

    def test_deps_requirements_parsing(self):
        """Basic requirements.txt produces a chunk listing packages."""
        repo = _make_repo({
            "requirements.txt": (
                "# comment\n"
                "requests>=2.28\n"
                "fastapi[all]==0.100.0\n"
                "uvicorn\n"
                "-r other.txt\n"
            )
        })
        # mock pip list so we don't actually call pip
        with patch("collect_context._run", return_value=(False, "")):
            chunks = deps_context(repo)

        req_chunk = next(c for c in chunks if "requirements" in c["id"])
        self.assertEqual(req_chunk["type"], "env_deps")
        self.assertEqual(req_chunk["language"], "environment")
        self.assertIn("requests", req_chunk["text"])
        self.assertIn("fastapi", req_chunk["text"])
        self.assertIn("uvicorn", req_chunk["text"])
        self.assertEqual(req_chunk["metadata"]["collector"], "deps")


# ---------------------------------------------------------------------------
# 2. test_deps_diff_missing_package
# ---------------------------------------------------------------------------

class TestDepsDiffMissingPackage(unittest.TestCase):

    def test_deps_diff_missing_package(self):
        """Package in requirements but not in pip list -> flagged as MISSING."""
        repo = _make_repo({
            "requirements.txt": "requests\nnon-existent-pkg\n"
        })

        fake_pip_json = json.dumps([
            {"name": "requests", "version": "2.28.0"},
            {"name": "pip", "version": "23.0"},
        ])

        call_count = [0]

        def _controlled_run(cmd, timeout=10, cwd=None):
            call_count[0] += 1
            if "pip" in cmd and "list" in cmd:
                return True, fake_pip_json
            return False, ""

        with patch("collect_context._run", side_effect=_controlled_run):
            chunks = deps_context(repo)

        diff_chunk = next((c for c in chunks if "pip-diff" in c["id"]), None)
        self.assertIsNotNone(diff_chunk, "Expected pip-diff chunk")
        self.assertIn("MISSING", diff_chunk["text"])
        self.assertIn("non-existent-pkg", diff_chunk["text"])
        self.assertIn("non-existent-pkg", diff_chunk["metadata"]["missing"])


# ---------------------------------------------------------------------------
# 3. test_deps_diff_undeclared_package
# ---------------------------------------------------------------------------

class TestDepsDiffUndeclaredPackage(unittest.TestCase):

    def test_deps_diff_undeclared_package(self):
        """Package installed but not in requirements -> flagged as UNDECLARED."""
        repo = _make_repo({
            "requirements.txt": "requests\n"
        })

        fake_pip_json = json.dumps([
            {"name": "requests", "version": "2.28.0"},
            {"name": "black", "version": "23.0"},      # not declared
            {"name": "mypy", "version": "1.0"},        # not declared
        ])

        def _controlled_run(cmd, timeout=10, cwd=None):
            if "pip" in cmd and "list" in cmd:
                return True, fake_pip_json
            return False, ""

        with patch("collect_context._run", side_effect=_controlled_run):
            chunks = deps_context(repo)

        diff_chunk = next((c for c in chunks if "pip-diff" in c["id"]), None)
        self.assertIsNotNone(diff_chunk, "Expected pip-diff chunk")
        self.assertIn("UNDECLARED", diff_chunk["text"])
        undeclared = diff_chunk["metadata"]["undeclared"]
        self.assertIn("black", undeclared)
        self.assertIn("mypy", undeclared)


# ---------------------------------------------------------------------------
# 4. test_git_history_chunk_format
# ---------------------------------------------------------------------------

class TestGitHistoryChunkFormat(unittest.TestCase):

    def test_git_history_chunk_format(self):
        """Fake git log output produces a well-structured git_history chunk."""
        fake_log = (
            "abc1234 feat: add authentication module\n"
            "def5678 fix: resolve null pointer in parser\n"
            "ghi9012 docs: update README\n"
        )
        fake_shortlog = "  5  Alice Smith\n  3  Bob Jones\n"

        def _controlled_run(cmd, timeout=10, cwd=None):
            if "log" in cmd:
                return True, fake_log
            if "shortlog" in cmd:
                return True, fake_shortlog
            if "show-current" in cmd or "branch" in cmd:
                return True, "main"
            if "status" in cmd:
                return True, "M  app/core/security.py"
            return False, ""

        repo = _make_repo({})
        with patch("collect_context._run", side_effect=_controlled_run):
            chunks = git_context(repo)

        history_chunk = next(c for c in chunks if c["type"] == "git_history")
        self.assertIn("branch: main", history_chunk["text"])
        self.assertIn("abc1234", history_chunk["text"])
        self.assertIn("Alice Smith", history_chunk["text"])
        self.assertEqual(history_chunk["metadata"]["branch"], "main")
        self.assertEqual(history_chunk["metadata"]["commit_count"], 3)


# ---------------------------------------------------------------------------
# 5. test_env_redaction_secret
# ---------------------------------------------------------------------------

class TestEnvRedactionSecret(unittest.TestCase):

    def test_env_redaction_secret(self):
        """Variable with SECRET in its name has value replaced with <redacted>."""
        self.assertEqual(_redact_value("MY_SECRET", "supersecret123"), "<redacted>")
        self.assertEqual(_redact_value("DB_SECRET_KEY", "abc"), "<redacted>")

    def test_env_redaction_is_case_insensitive(self):
        self.assertEqual(_redact_value("my_secret", "val"), "<redacted>")
        self.assertEqual(_redact_value("Secret_Token", "val"), "<redacted>")

    def test_env_redaction_in_chunk(self):
        """env_context chunk must not expose secret values."""
        repo = _make_repo({})
        fake_env = {
            "MY_APP_SECRET": "should-be-hidden",
            "APP_NAME": "myapp",
            "SOME_PATH": "/usr/local",
        }
        with patch.dict("os.environ", fake_env, clear=False):
            chunks = env_context(repo)

        text = chunks[0]["text"]
        self.assertIn("MY_APP_SECRET=<redacted>", text)
        self.assertNotIn("should-be-hidden", text)


# ---------------------------------------------------------------------------
# 6. test_env_redaction_token
# ---------------------------------------------------------------------------

class TestEnvRedactionToken(unittest.TestCase):

    def test_env_redaction_token(self):
        """Variable with TOKEN in its name must be redacted."""
        self.assertEqual(_redact_value("GITHUB_TOKEN", "ghp_123abc"), "<redacted>")
        self.assertEqual(_redact_value("ACCESS_TOKEN_SECRET", "x"), "<redacted>")

    def test_env_redaction_token_in_chunk(self):
        repo = _make_repo({})
        fake_env = {"API_TOKEN": "tok_live_xyz789"}
        with patch.dict("os.environ", fake_env, clear=False):
            chunks = env_context(repo)
        text = chunks[0]["text"]
        self.assertIn("API_TOKEN=<redacted>", text)
        self.assertNotIn("tok_live_xyz789", text)


# ---------------------------------------------------------------------------
# 7. test_env_no_redaction_normal
# ---------------------------------------------------------------------------

class TestEnvNoRedactionNormal(unittest.TestCase):

    def test_env_no_redaction_normal(self):
        """Normal variable (no sensitive keyword) -> value is NOT redacted."""
        self.assertEqual(_redact_value("APP_ENV", "production"), "production")
        self.assertEqual(_redact_value("LOG_LEVEL", "DEBUG"), "DEBUG")

    def test_normal_var_appears_in_chunk(self):
        repo = _make_repo({})
        fake_env = {"MY_APP_ENV": "staging"}
        with patch.dict("os.environ", fake_env, clear=False):
            chunks = env_context(repo)
        text = chunks[0]["text"]
        self.assertIn("MY_APP_ENV=staging", text)


# ---------------------------------------------------------------------------
# 8. test_docker_no_docker
# ---------------------------------------------------------------------------

class TestDockerNoDocket(unittest.TestCase):

    def test_docker_no_docker(self):
        """When docker is unavailable, chunk contains graceful message."""
        repo = _make_repo({})
        with patch("collect_context._run", return_value=(False, "")):
            chunks = docker_context(repo)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "docker_state")
        text = chunks[0]["text"].lower()
        self.assertTrue(
            "not available" in text or "no containers" in text,
            f"Expected graceful fallback message, got: {chunks[0]['text']!r}"
        )

    def test_docker_empty_output_graceful(self):
        """Empty docker ps output also produces graceful message."""
        repo = _make_repo({})
        with patch("collect_context._run", return_value=(True, "")):
            chunks = docker_context(repo)
        self.assertEqual(len(chunks), 1)
        text = chunks[0]["text"].lower()
        self.assertTrue("not available" in text or "no containers" in text)


# ---------------------------------------------------------------------------
# 9. test_docker_compose_diff
# ---------------------------------------------------------------------------

class TestDockerComposeDiff(unittest.TestCase):

    def test_docker_compose_diff(self):
        """Services declared in docker-compose.yml but not running are flagged."""
        compose_content = (
            "version: '3.8'\n"
            "services:\n"
            "  web:\n"
            "    image: nginx:alpine\n"
            "  db:\n"
            "    image: postgres:14\n"
            "  redis:\n"
            "    image: redis:7\n"
        )
        repo = _make_repo({"docker-compose.yml": compose_content})

        # Only 'web' is running
        def _controlled_run(cmd, timeout=10, cwd=None):
            if "docker" in cmd and "ps" in cmd:
                return True, "web\tnginx:alpine\tUp 2 hours\t0.0.0.0:80->80/tcp"
            return False, ""

        with patch("collect_context._run", side_effect=_controlled_run):
            chunks = docker_context(repo)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        text = chunk["text"]
        self.assertIn("web", text)
        self.assertIn("RUNNING", text)
        self.assertIn("NOT RUNNING", text)

        meta = chunk["metadata"]
        not_running = set(meta["not_running"])
        self.assertIn("db", not_running)
        self.assertIn("redis", not_running)
        self.assertNotIn("web", not_running)

    def test_parse_compose_services(self):
        """_parse_compose_services extracts service names correctly."""
        compose = (
            "version: '3'\n"
            "services:\n"
            "  alpha:\n"
            "    image: x\n"
            "  beta:\n"
            "    image: y\n"
            "volumes:\n"
            "  mydata:\n"
        )
        services = _parse_compose_services(compose)
        self.assertIn("alpha", services)
        self.assertIn("beta", services)
        self.assertNotIn("mydata", services)


# ---------------------------------------------------------------------------
# 10. test_process_filter_relevant
# ---------------------------------------------------------------------------

class TestProcessFilterRelevant(unittest.TestCase):

    def test_process_filter_relevant_windows(self):
        """Windows tasklist CSV output: only relevant processes are kept."""
        fake_tasklist = (
            '"uvicorn.exe","1234","Console","1","45,000 K"\n'
            '"python.exe","5678","Console","1","120,000 K"\n'
            '"notepad.exe","9999","Console","1","5,000 K"\n'
            '"svchost.exe","111","Services","0","10,000 K"\n'
        )

        def _controlled_run(cmd, timeout=10, cwd=None):
            if "tasklist" in cmd:
                return True, fake_tasklist
            return False, ""

        repo = _make_repo({})
        with patch("collect_context._run", side_effect=_controlled_run):
            with patch("collect_context.platform.system", return_value="Windows"):
                chunks = process_context(repo)

        self.assertEqual(len(chunks), 1)
        text = chunks[0]["text"]
        self.assertIn("uvicorn", text.lower())
        self.assertIn("python", text.lower())
        self.assertNotIn("notepad", text.lower())
        self.assertNotIn("svchost", text.lower())
        self.assertEqual(chunks[0]["metadata"]["process_count"], 2)

    def test_process_no_relevant_processes(self):
        """When no relevant processes are running, chunk says so gracefully."""
        fake_tasklist = (
            '"notepad.exe","9999","Console","1","5,000 K"\n'
        )

        def _controlled_run(cmd, timeout=10, cwd=None):
            if "tasklist" in cmd:
                return True, fake_tasklist
            return False, ""

        repo = _make_repo({})
        with patch("collect_context._run", side_effect=_controlled_run):
            with patch("collect_context.platform.system", return_value="Windows"):
                chunks = process_context(repo)

        text = chunks[0]["text"].lower()
        self.assertIn("no relevant", text)


# ---------------------------------------------------------------------------
# 11. test_test_results_lastfailed
# ---------------------------------------------------------------------------

class TestTestResultsLastFailed(unittest.TestCase):

    def test_test_results_lastfailed(self):
        """.pytest_cache/lastfailed is parsed and failed tests appear in chunk."""
        lastfailed_data = {
            "tests/test_auth.py::test_login_invalid_password": True,
            "tests/test_user.py::test_create_user_no_email": True,
        }
        repo = _make_repo({
            ".pytest_cache/lastfailed": json.dumps(lastfailed_data),
        })

        with patch("collect_context._run", return_value=(False, "")):
            chunks = testresults_context(repo)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["type"], "test_results")
        text = chunk["text"]
        self.assertIn("test_login_invalid_password", text)
        self.assertIn("test_create_user_no_email", text)
        self.assertIn("Failed tests", text)

        meta = chunk["metadata"]
        self.assertIn("tests/test_auth.py::test_login_invalid_password", meta["failed_tests"])

    def test_test_results_junit_xml(self):
        """JUnit XML is parsed to extract pass/fail counts."""
        junit_xml = (
            '<?xml version="1.0"?>\n'
            '<testsuite name="pytest" tests="10" failures="2" errors="1" skipped="0">\n'
            '  <testcase classname="test_auth" name="test_login_ok"/>\n'
            '  <testcase classname="test_auth" name="test_login_fail">'
            '<failure>AssertionError</failure></testcase>\n'
            '  <testcase classname="test_user" name="test_create">'
            '<error>RuntimeError</error></testcase>\n'
            '</testsuite>\n'
        )
        repo = _make_repo({"pytest.xml": junit_xml})

        with patch("collect_context._run", return_value=(False, "")):
            chunks = testresults_context(repo)

        meta = chunks[0]["metadata"]
        self.assertEqual(meta["failures"], 2)
        self.assertEqual(meta["errors"], 1)

    def test_coverage_xml_parsed(self):
        """coverage.xml line-rate is extracted correctly."""
        cov_xml = (
            '<?xml version="1.0"?>\n'
            '<coverage version="7.0" line-rate="0.83">\n'
            '</coverage>\n'
        )
        repo = _make_repo({"coverage.xml": cov_xml})

        with patch("collect_context._run", return_value=(False, "")):
            chunks = testresults_context(repo)

        meta = chunks[0]["metadata"]
        self.assertEqual(meta["coverage_pct"], 83.0)
        self.assertIn("83.0%", chunks[0]["text"])


# ---------------------------------------------------------------------------
# 12. test_chunk_ids_no_backslash
# ---------------------------------------------------------------------------

class TestChunkIdsNoBackslash(unittest.TestCase):

    def test_chunk_ids_no_backslash(self):
        """All chunk IDs produced by collect_all must use forward slashes only."""
        repo = _make_repo({
            "requirements.txt": "requests\n",
        })

        with patch("collect_context._run", return_value=(False, "")):
            chunks = collect_all(repo_path=repo, only=["deps", "git", "env", "docker", "process", "tests"])

        for chunk in chunks:
            self.assertNotIn("\\", chunk["id"],
                             f"Backslash in chunk ID: {chunk['id']!r}")
            # ID must contain "::"
            self.assertIn("::", chunk["id"],
                          f"Missing '::' separator in chunk ID: {chunk['id']!r}")

    def test_chunk_ids_max_length(self):
        """All chunk IDs must be at most 80 characters."""
        repo = _make_repo({})
        with patch("collect_context._run", return_value=(False, "")):
            chunks = collect_all(repo_path=repo)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["id"]), 80,
                                 f"Chunk ID too long ({len(chunk['id'])} chars): {chunk['id']!r}")


# ---------------------------------------------------------------------------
# 13. test_merge_with_positions_context_first
# ---------------------------------------------------------------------------

class TestMergeWithPositionsContextFirst(unittest.TestCase):

    def test_merge_with_context_after_overview(self):
        """Context chunks appear after repo_overview and before other chunks."""
        overview_chunk = {
            "id": "repo_overview",
            "type": "repo_overview",
            "language": "python",
            "text": "Repository root: myrepo",
            "metadata": {},
        }
        code_chunk_1 = {
            "id": "file::app/main.py",
            "type": "file",
            "language": "python",
            "text": "File: app/main.py",
            "metadata": {},
        }
        code_chunk_2 = {
            "id": "function::app/main.py::main",
            "type": "function",
            "language": "python",
            "text": "Function: main",
            "metadata": {},
        }

        existing_jsonl = "\n".join(
            json.dumps(c) for c in [overview_chunk, code_chunk_1, code_chunk_2]
        )

        repo = _make_repo({})

        import os
        tmp_merge = Path(tempfile.mktemp(suffix=".jsonl"))
        tmp_merge.write_text(existing_jsonl, encoding="utf-8")
        tmp_out = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            with patch("collect_context._run", return_value=(False, "")):
                collect_all(
                    repo_path=repo,
                    only=["git"],
                    output_path=tmp_out,
                    merge_with=tmp_merge,
                )

            result = []
            for line in tmp_out.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    result.append(json.loads(line))

            # First chunk must be repo_overview
            self.assertEqual(result[0]["id"], "repo_overview")

            # Context chunks (git_history, git_status) must come before code chunks
            ids = [c["id"] for c in result]
            git_history_idx = next(i for i, c in enumerate(result) if c["type"] == "git_history")
            file_chunk_idx = next(i for i, c in enumerate(result) if c["id"] == "file::app/main.py")
            self.assertLess(git_history_idx, file_chunk_idx,
                            "git_history chunk should appear before file chunks")
        finally:
            for f in (tmp_merge, tmp_out):
                try:
                    f.unlink()
                except Exception:
                    pass

    def test_merge_without_overview(self):
        """When no repo_overview exists, context chunks are prepended."""
        code_chunk = {
            "id": "file::main.py",
            "type": "file",
            "language": "python",
            "text": "File: main.py",
            "metadata": {},
        }
        existing_jsonl = json.dumps(code_chunk)

        repo = _make_repo({})
        tmp_merge = Path(tempfile.mktemp(suffix=".jsonl"))
        tmp_merge.write_text(existing_jsonl, encoding="utf-8")
        tmp_out = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            with patch("collect_context._run", return_value=(False, "")):
                collect_all(
                    repo_path=repo,
                    only=["env"],
                    output_path=tmp_out,
                    merge_with=tmp_merge,
                )

            result = []
            for line in tmp_out.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    result.append(json.loads(line))

            # env chunk should come before the file chunk (no overview to anchor after)
            ids = [c["id"] for c in result]
            env_idx = next(i for i, c in enumerate(result) if c["type"] == "env_vars")
            file_idx = next(i for i, c in enumerate(result) if c["id"] == "file::main.py")
            self.assertLess(env_idx, file_idx,
                            "env_vars chunk should appear before file chunk when no overview")
        finally:
            for f in (tmp_merge, tmp_out):
                try:
                    f.unlink()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Bonus: collector isolation -- one failing collector does not break others
# ---------------------------------------------------------------------------

class TestCollectorIsolation(unittest.TestCase):

    def test_failing_collector_does_not_block_others(self):
        """If one collector raises, the rest still run."""
        repo = _make_repo({})

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated collector failure")

        import collect_context as cc
        original_git = cc.git_context
        cc.git_context = _boom
        try:
            with patch("collect_context._run", return_value=(False, "")):
                # Should not raise; other collectors still produce chunks
                chunks = collect_all(repo_path=repo, only=["git", "env", "process"])
            # env and process chunks should still be present
            types = {c["type"] for c in chunks}
            self.assertTrue(
                "env_vars" in types or "process_state" in types,
                "Expected at least env_vars or process_state despite git failure"
            )
        finally:
            cc.git_context = original_git


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
