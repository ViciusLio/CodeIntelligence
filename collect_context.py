"""
collect_context.py -- Collect structured environment context as JSONL chunks.

Produces env/runtime context chunks in the same format as parse_repo.py,
ready to be merged with code chunks for RAG retrieval.

Usage:
    python collect_context.py --repo /path/to/repo --output context_chunks.jsonl
    python collect_context.py --repo . --only deps,git,docker --output ctx.jsonl
    python collect_context.py --repo . --merge-with repo_chunks.jsonl --output full.jsonl

Collectors:
    deps    -- dependency files + pip diff (env_deps)
    git     -- recent history + status (git_history, git_status)
    env     -- environment variables, redacted (env_vars)
    docker  -- running containers + compose diff (docker_state)
    process -- relevant running processes (process_state)
    tests   -- last test results + coverage (test_results)

Output format (one JSON object per line):
    {
      "id":       "env_deps::requirements",
      "type":     "env_deps" | "git_history" | "git_status" | "env_vars"
                  | "docker_state" | "process_state" | "test_results",
      "language": "environment",
      "text":     str,
      "metadata": dict
    }

Zero external dependencies -- stdlib only.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERN = re.compile(
    r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH|PRIVATE|CERT|SIGNING)",
    re.IGNORECASE,
)

# Windows system env vars to filter out
_WIN_SYSTEM_VARS = {
    "ALLUSERSPROFILE", "APPDATA", "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432", "COMPUTERNAME",
    "COMSPEC", "DRIVERDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA",
    "LOGONSERVER", "NUMBER_OF_PROCESSORS", "OS", "PATHEXT",
    "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "PROGRAMW6432", "PSMODULEPATH", "PUBLIC", "SESSIONNAME",
    "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "USERDOMAIN",
    "USERDOMAIN_ROAMINGPROFILE", "USERNAME", "USERPROFILE", "WINDIR",
    "ONEDRIVE", "ONEDRIVECONSUMER",
}

def _run(cmd: list[str], timeout: int = 10, cwd: str | None = None) -> tuple[bool, str]:
    """Run a subprocess safely. Returns (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return True, result.stdout.strip()
    except FileNotFoundError:
        return False, ""
    except subprocess.TimeoutExpired:
        return False, ""
    except Exception:
        return False, ""


def _make_chunk(
    chunk_id: str,
    chunk_type: str,
    text: str,
    metadata: dict,
) -> dict:
    return {
        "id": chunk_id,
        "type": chunk_type,
        "language": "environment",
        "text": text,
        "metadata": metadata,
    }


def _redact_value(name: str, value: str) -> str:
    """Return '<redacted>' if name contains a sensitive keyword, else the value."""
    if _SENSITIVE_PATTERN.search(name):
        return "<redacted>"
    return value


# ---------------------------------------------------------------------------
# Collector 1: deps_context
# ---------------------------------------------------------------------------

_STDLIB_LIKE = {
    "pip", "setuptools", "wheel", "pkg-resources", "pkg_resources",
    "distribute", "distlib", "platformdirs", "packaging",
}


def _parse_requirements_txt(text: str) -> list[str]:
    """Return list of package names from requirements.txt content."""
    pkgs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # strip extras / version specs
        name = re.split(r"[>=<!;\[\s]", line)[0].strip()
        if name:
            pkgs.append(name.lower())
    return pkgs


def _parse_package_json(text: str) -> list[str]:
    """Return deps + devDeps from package.json."""
    try:
        data = json.loads(text)
    except Exception:
        return []
    deps = list(data.get("dependencies", {}).keys())
    deps += list(data.get("devDependencies", {}).keys())
    return [d.lower() for d in deps]


def _parse_go_mod(text: str) -> list[str]:
    """Return module names from go.mod require block."""
    in_require = False
    pkgs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("require ("):
            in_require = True
            continue
        if in_require and line == ")":
            in_require = False
            continue
        if in_require or line.startswith("require "):
            parts = line.replace("require ", "").split()
            if parts:
                pkgs.append(parts[0].lower())
    return pkgs


def _parse_gemfile(text: str) -> list[str]:
    pkgs = []
    for line in text.splitlines():
        m = re.match(r"\s*gem\s+['\"]([^'\"]+)['\"]", line)
        if m:
            pkgs.append(m.group(1).lower())
    return pkgs


def _parse_cargo_toml(text: str) -> list[str]:
    pkgs = []
    in_deps = False
    for line in text.splitlines():
        line = line.strip()
        if line == "[dependencies]":
            in_deps = True
            continue
        if line.startswith("[") and line != "[dependencies]":
            in_deps = False
            continue
        if in_deps:
            m = re.match(r'^([A-Za-z0-9_-]+)\s*=', line)
            if m:
                pkgs.append(m.group(1).lower())
    return pkgs


def _parse_pyproject_toml(text: str) -> list[str]:
    """Extract deps from pyproject.toml [project.dependencies] or [tool.poetry.dependencies]."""
    pkgs = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in ("[project.dependencies]", "[tool.poetry.dependencies]"):
            in_section = True
            continue
        if in_section and stripped.startswith("["):
            in_section = False
            continue
        if in_section:
            # PEP 508 style: "requests>=2.0" or poetry: requests = "^2.0"
            name = re.split(r"[>=<!\[\s\"]", stripped)[0].strip()
            if name and not name.startswith("#"):
                pkgs.append(name.lower())
    return pkgs


def deps_context(repo: Path) -> list[dict]:
    """Collect dependency files + pip diff."""
    chunks: list[dict] = []
    declared_packages: list[str] = []

    dep_files = [
        ("requirements.txt", "requirements"),
        ("requirements-dev.txt", "requirements-dev"),
        ("requirements-test.txt", "requirements-test"),
        ("pyproject.toml", "pyproject"),
        ("package.json", "package-json"),
        ("go.mod", "go-mod"),
        ("Gemfile", "gemfile"),
        ("Cargo.toml", "cargo-toml"),
    ]

    for filename, slug in dep_files:
        fpath = repo / filename
        if not fpath.exists():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if filename.startswith("requirements"):
            pkgs = _parse_requirements_txt(text)
            declared_packages.extend(pkgs)
            chunk_text = (
                f"Dependency file: {filename}\n"
                f"Packages declared ({len(pkgs)}):\n"
                + "\n".join(f"  - {p}" for p in pkgs)
            )
        elif filename == "pyproject.toml":
            pkgs = _parse_pyproject_toml(text)
            declared_packages.extend(pkgs)
            chunk_text = (
                f"Dependency file: {filename}\n"
                f"Python dependencies ({len(pkgs)}):\n"
                + "\n".join(f"  - {p}" for p in pkgs)
            )
        elif filename == "package.json":
            pkgs = _parse_package_json(text)
            chunk_text = (
                f"Dependency file: {filename}\n"
                f"JS/Node packages ({len(pkgs)}):\n"
                + "\n".join(f"  - {p}" for p in pkgs[:50])
            )
        elif filename == "go.mod":
            pkgs = _parse_go_mod(text)
            chunk_text = (
                f"Dependency file: {filename}\n"
                f"Go modules ({len(pkgs)}):\n"
                + "\n".join(f"  - {p}" for p in pkgs[:50])
            )
        elif filename == "Gemfile":
            pkgs = _parse_gemfile(text)
            chunk_text = (
                f"Dependency file: {filename}\n"
                f"Ruby gems ({len(pkgs)}):\n"
                + "\n".join(f"  - {p}" for p in pkgs[:50])
            )
        elif filename == "Cargo.toml":
            pkgs = _parse_cargo_toml(text)
            chunk_text = (
                f"Dependency file: {filename}\n"
                f"Rust crates ({len(pkgs)}):\n"
                + "\n".join(f"  - {p}" for p in pkgs[:50])
            )
        else:
            pkgs = []
            chunk_text = f"Dependency file: {filename}\n{text[:500]}"

        chunks.append(_make_chunk(
            f"env_deps::{slug}",
            "env_deps",
            chunk_text,
            {
                "collector": "deps",
                "file": filename,
                "package_count": len(pkgs),
            },
        ))

    # pip list diff (Python only)
    ok, pip_out = _run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        timeout=15,
    )
    if ok and pip_out:
        try:
            pip_pkgs_raw = json.loads(pip_out)
            installed = {p["name"].lower() for p in pip_pkgs_raw}

            declared_set = set(declared_packages)
            missing = sorted(declared_set - installed - _STDLIB_LIKE)
            undeclared_raw = installed - declared_set - _STDLIB_LIKE
            # filter common stdlib-like & keep top 20 by name
            undeclared = sorted(
                p for p in undeclared_raw
                if p not in _STDLIB_LIKE
            )[:20]

            diff_lines = [
                f"Python environment diff (installed vs declared):",
                f"  Installed packages: {len(installed)}",
                f"  Declared in requirements: {len(declared_set)}",
            ]
            if missing:
                diff_lines.append(f"\n  MISSING (in requirements but not installed):")
                diff_lines.extend(f"    - {p}" for p in missing)
            if undeclared:
                diff_lines.append(f"\n  UNDECLARED (installed but not in requirements, top 20):")
                diff_lines.extend(f"    - {p}" for p in undeclared)
            if not missing and not undeclared:
                diff_lines.append("  All declared packages are installed. No undeclared extras detected.")

            chunks.append(_make_chunk(
                "env_deps::pip-diff",
                "env_deps",
                "\n".join(diff_lines),
                {
                    "collector": "deps",
                    "installed_count": len(installed),
                    "declared_count": len(declared_set),
                    "missing": missing,
                    "undeclared": undeclared,
                },
            ))
        except Exception:
            pass

    return chunks


# ---------------------------------------------------------------------------
# Collector 2: git_context
# ---------------------------------------------------------------------------

def git_context(repo: Path) -> list[dict]:
    """Collect git history + status chunks."""
    chunks: list[dict] = []
    repo_str = str(repo)

    ok_branch, branch = _run(
        ["git", "branch", "--show-current"],
        cwd=repo_str,
    )
    branch = branch.strip() if ok_branch else "unknown"

    ok_log, log_out = _run(
        ["git", "log", "--oneline", "-20"],
        cwd=repo_str,
    )

    ok_shortlog, shortlog_out = _run(
        ["git", "shortlog", "-sn", "--since=30 days ago"],
        cwd=repo_str,
    )

    if ok_log:
        commit_lines = [l for l in log_out.splitlines() if l.strip()]
        commit_count = len(commit_lines)

        history_lines = [
            f"Repository git context (branch: {branch})",
            "",
            f"Recent commits (last {commit_count}):",
        ]
        for line in commit_lines:
            history_lines.append(f"  {line}")

        if ok_shortlog and shortlog_out.strip():
            history_lines.append("")
            history_lines.append("Active contributors (last 30 days):")
            for line in shortlog_out.splitlines():
                if line.strip():
                    history_lines.append(f"  {line.strip()}")

        chunks.append(_make_chunk(
            "git_history::recent",
            "git_history",
            "\n".join(history_lines),
            {
                "collector": "git",
                "branch": branch,
                "commit_count": commit_count,
            },
        ))
    else:
        chunks.append(_make_chunk(
            "git_history::recent",
            "git_history",
            f"Repository git context (branch: {branch})\n\nGit not available or not a git repository.",
            {
                "collector": "git",
                "branch": branch,
                "commit_count": 0,
            },
        ))

    ok_status, status_out = _run(
        ["git", "status", "--short"],
        cwd=repo_str,
    )

    if ok_status:
        status_lines = [l for l in status_out.splitlines() if l.strip()]
        text_lines = ["Modified files (git status --short):"]
        if status_lines:
            for line in status_lines:
                text_lines.append(f"  {line}")
        else:
            text_lines.append("  (working tree clean)")

        chunks.append(_make_chunk(
            "git_status::current",
            "git_status",
            "\n".join(text_lines),
            {
                "collector": "git",
                "branch": branch,
                "modified_count": len(status_lines),
            },
        ))
    else:
        chunks.append(_make_chunk(
            "git_status::current",
            "git_status",
            "Modified files (git status --short):\n  Git not available or not a git repository.",
            {
                "collector": "git",
                "branch": branch,
                "modified_count": 0,
            },
        ))

    return chunks


# ---------------------------------------------------------------------------
# Collector 3: env_context
# ---------------------------------------------------------------------------

def env_context(repo: Path) -> list[dict]:
    """Collect environment variables (redacted for secrets)."""
    chunks: list[dict] = []

    all_vars = dict(os.environ)

    # Filter out Windows system noise
    custom_vars = {
        k: v for k, v in all_vars.items()
        if k.upper() not in _WIN_SYSTEM_VARS
        and k not in ("PATH", "PYTHONPATH", "PATHEXT")
    }

    lines = ["Environment variables (sensitive values redacted):"]
    redacted_count = 0
    included_count = 0

    for name, value in sorted(custom_vars.items()):
        display_val = _redact_value(name, value)
        if display_val == "<redacted>":
            redacted_count += 1
            lines.append(f"  {name}=<redacted>")
        else:
            if len(value) < 200:
                lines.append(f"  {name}={value}")
                included_count += 1

    # Also read .env.example and .env.local if present
    extra_files = [".env.example", ".env.local"]
    for fname in extra_files:
        fpath = repo / fname
        if not fpath.exists():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines.append(f"\nFrom {fname}:")
        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue
            if "=" in raw_line:
                eq_idx = raw_line.index("=")
                var_name = raw_line[:eq_idx].strip()
                var_val = raw_line[eq_idx + 1:].strip()
                display_val = _redact_value(var_name, var_val)
                if display_val == "<redacted>":
                    lines.append(f"  {var_name}=<redacted>")
                else:
                    if len(var_val) < 200:
                        lines.append(f"  {var_name}={var_val}")

    chunks.append(_make_chunk(
        "env_vars::current",
        "env_vars",
        "\n".join(lines),
        {
            "collector": "env",
            "total_vars": len(custom_vars),
            "redacted_count": redacted_count,
            "included_count": included_count,
        },
    ))
    return chunks


# ---------------------------------------------------------------------------
# Collector 4: docker_context
# ---------------------------------------------------------------------------

def _parse_compose_services(text: str) -> list[str]:
    """Extract service names from docker-compose.yml (no YAML library needed)."""
    services: list[str] = []
    in_services = False
    for line in text.splitlines():
        # Detect 'services:' at root level (no leading spaces)
        if re.match(r"^services\s*:", line):
            in_services = True
            continue
        if in_services:
            # End of services block: another root-level key
            if re.match(r"^[a-zA-Z]", line) and not line.startswith(" "):
                in_services = False
                continue
            # Service name: exactly 2-space indent + key
            m = re.match(r"^  ([a-zA-Z0-9_-]+)\s*:", line)
            if m:
                services.append(m.group(1))
    return services


def docker_context(repo: Path) -> list[dict]:
    """Collect Docker container state + compose diff."""
    chunks: list[dict] = []

    ok, docker_out = _run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"],
        timeout=10,
    )

    if not ok or not docker_out.strip():
        message = (
            "Docker state:\n"
            "  Docker not available or no containers running."
        )
        running_containers: list[str] = []
    else:
        rows = [l for l in docker_out.splitlines() if l.strip()]
        running_containers = []
        lines = ["Docker running containers:"]
        for row in rows:
            parts = row.split("\t")
            name = parts[0] if len(parts) > 0 else "?"
            image = parts[1] if len(parts) > 1 else "?"
            status = parts[2] if len(parts) > 2 else "?"
            ports = parts[3] if len(parts) > 3 else ""
            running_containers.append(name)
            entry = f"  {name}  [{image}]  {status}"
            if ports:
                entry += f"  ports: {ports}"
            lines.append(entry)
        message = "\n".join(lines)

    # Check docker-compose files
    compose_path: Path | None = None
    for fname in ("docker-compose.yml", "docker-compose.yaml"):
        candidate = repo / fname
        if candidate.exists():
            compose_path = candidate
            break

    declared_services: list[str] = []
    if compose_path is not None:
        try:
            compose_text = compose_path.read_text(encoding="utf-8", errors="replace")
            declared_services = _parse_compose_services(compose_text)
        except Exception:
            pass

    if declared_services:
        running_set = set(running_containers)
        declared_set = set(declared_services)
        not_running = sorted(declared_set - running_set)

        message += f"\n\nDocker Compose declared services ({len(declared_services)}):"
        for svc in declared_services:
            status_label = "RUNNING" if svc in running_set else "NOT RUNNING"
            message += f"\n  {svc}  [{status_label}]"

        if not_running:
            message += f"\n\nServices declared but not running: {', '.join(not_running)}"

    chunks.append(_make_chunk(
        "docker_state::current",
        "docker_state",
        message,
        {
            "collector": "docker",
            "running_containers": running_containers,
            "declared_services": declared_services,
            "not_running": sorted(set(declared_services) - set(running_containers)),
        },
    ))
    return chunks


# ---------------------------------------------------------------------------
# Collector 5: process_context
# ---------------------------------------------------------------------------

_RELEVANT_PROCESS_NAMES = {
    "python", "uvicorn", "gunicorn", "celery", "redis",
    "postgres", "nginx", "node", "docker", "ollama",
}


def process_context(repo: Path) -> list[dict]:  # noqa: ARG001  repo unused but kept for API consistency
    """Collect relevant running processes."""
    system = platform.system()

    if system == "Windows":
        ok, raw = _run(["tasklist", "/FO", "CSV", "/NH"], timeout=10)
        processes: list[dict] = []
        if ok and raw:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                # CSV format: "python.exe","12345","Console","1","50,000 K"
                parts = [p.strip('"') for p in line.split('","')]
                if len(parts) < 5:
                    continue
                pname = parts[0].lower()
                pid = parts[1]
                mem = parts[4]
                base = pname.replace(".exe", "")
                if any(rel in base for rel in _RELEVANT_PROCESS_NAMES):
                    processes.append({"name": parts[0], "pid": pid, "mem": mem})
    else:
        ok, raw = _run(["ps", "aux"], timeout=10)
        processes = []
        if ok and raw:
            for line in raw.splitlines()[1:]:  # skip header
                parts = line.split()
                if len(parts) < 11:
                    continue
                pname = parts[10].lower()
                pid = parts[1]
                mem = parts[5]
                if any(rel in pname for rel in _RELEVANT_PROCESS_NAMES):
                    processes.append({"name": parts[10], "pid": pid, "mem": mem})

    if processes:
        lines = [f"Relevant running processes (OS: {system}):"]
        for p in processes:
            lines.append(f"  {p['name']}  PID={p['pid']}  mem={p['mem']}")
        text = "\n".join(lines)
    else:
        text = f"Relevant running processes (OS: {system}):\n  No relevant processes found."

    return [_make_chunk(
        "process_state::current",
        "process_state",
        text,
        {
            "collector": "process",
            "os": system,
            "process_count": len(processes),
            "processes": processes,
        },
    )]


# ---------------------------------------------------------------------------
# Collector 6: test_context
# ---------------------------------------------------------------------------

def _parse_junit_xml(xml_text: str) -> dict:
    """Parse JUnit XML and return summary dict."""
    try:
        root = ET.fromstring(xml_text)
        # Handle both <testsuite> and <testsuites><testsuite>...
        suites = []
        if root.tag == "testsuites":
            suites = list(root.findall("testsuite"))
        elif root.tag == "testsuite":
            suites = [root]

        tests = failures = errors = skipped = 0
        failed_names: list[str] = []

        for suite in suites:
            tests += int(suite.get("tests", 0))
            failures += int(suite.get("failures", 0))
            errors += int(suite.get("errors", 0))
            skipped += int(suite.get("skipped", 0))

            for tc in suite.findall(".//testcase"):
                if tc.find("failure") is not None or tc.find("error") is not None:
                    classname = tc.get("classname", "")
                    name = tc.get("name", "")
                    label = f"{classname}::{name}" if classname else name
                    failed_names.append(label)

        return {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "failed_names": failed_names,
        }
    except Exception:
        return {}


def _parse_coverage_xml(xml_text: str) -> float | None:
    """Return line-rate as percentage (0-100) or None."""
    try:
        root = ET.fromstring(xml_text)
        rate = root.get("line-rate")
        if rate is not None:
            return round(float(rate) * 100, 1)
    except Exception:
        pass
    return None


def testresults_context(repo: Path) -> list[dict]:
    """Collect test results + coverage info."""
    summary: dict = {}
    failed_tests: list[str] = []
    coverage_pct: float | None = None

    # 1. .pytest_cache/lastfailed
    lastfailed_path = repo / ".pytest_cache" / "lastfailed"
    if lastfailed_path.exists():
        try:
            data = json.loads(lastfailed_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                failed_tests = list(data.keys())
            elif isinstance(data, list):
                failed_tests = data
        except Exception:
            pass

    # 2. JUnit XML files
    junit_candidates = [
        repo / "pytest.xml",
        repo / "test-results.xml",
    ]
    for rep_dir in (repo / "reports", repo):
        for f in rep_dir.glob("junit*.xml") if rep_dir.is_dir() else []:
            junit_candidates.append(f)

    for jpath in junit_candidates:
        if jpath.exists():
            try:
                xml_text = jpath.read_text(encoding="utf-8", errors="replace")
                summary = _parse_junit_xml(xml_text)
                if summary:
                    # JUnit XML failed names take precedence over lastfailed
                    if summary.get("failed_names"):
                        failed_tests = summary["failed_names"]
                    break
            except Exception:
                pass

    # 3. coverage.xml
    cov_xml = repo / "coverage.xml"
    if cov_xml.exists():
        try:
            coverage_pct = _parse_coverage_xml(
                cov_xml.read_text(encoding="utf-8", errors="replace")
            )
        except Exception:
            pass

    # 4. .coverage -> python -m coverage report
    if coverage_pct is None and (repo / ".coverage").exists():
        ok, cov_out = _run(
            [sys.executable, "-m", "coverage", "report", "--format=total"],
            timeout=15,
            cwd=str(repo),
        )
        if ok and cov_out.strip():
            try:
                coverage_pct = float(cov_out.strip().rstrip("%"))
            except ValueError:
                # Older coverage: last line is "TOTAL ... XX%"
                for line in reversed(cov_out.splitlines()):
                    m = re.search(r"(\d+)%", line)
                    if m:
                        coverage_pct = float(m.group(1))
                        break

    # Build text
    passed = summary.get("tests", 0) - summary.get("failures", 0) - summary.get("errors", 0)
    lines = ["Test results summary:"]
    if summary:
        lines.append(
            f"  Last run: {passed} tests passed, "
            f"{summary.get('failures', 0)} failed, "
            f"{summary.get('errors', 0)} errors"
        )
        if summary.get("skipped"):
            lines[-1] += f", {summary['skipped']} skipped"
    else:
        lines.append("  No XML test results found.")

    if coverage_pct is not None:
        lines.append(f"  Coverage: {coverage_pct}%")

    if failed_tests:
        lines.append("")
        lines.append("  Failed tests:")
        for t in failed_tests[:20]:
            lines.append(f"    - {t}")
        if len(failed_tests) > 20:
            lines.append(f"    ... and {len(failed_tests) - 20} more")
    elif summary.get("failures", 0) == 0 and summary.get("errors", 0) == 0 and summary:
        lines.append("  All tests passed.")

    if not summary and not failed_tests and coverage_pct is None:
        lines.append("  No test result data found (run pytest to generate .pytest_cache or junit XML).")

    return [_make_chunk(
        "test_results::last-run",
        "test_results",
        "\n".join(lines),
        {
            "collector": "tests",
            "passed": passed if summary else None,
            "failures": summary.get("failures"),
            "errors": summary.get("errors"),
            "coverage_pct": coverage_pct,
            "failed_tests": failed_tests,
        },
    )]


# ---------------------------------------------------------------------------
# All collectors registry
# ---------------------------------------------------------------------------

_COLLECTORS = {
    "deps": deps_context,
    "git": git_context,
    "env": env_context,
    "docker": docker_context,
    "process": process_context,
    "tests": testresults_context,
}

_COLLECTOR_TYPES = {
    "deps": ["env_deps"],
    "git": ["git_history", "git_status"],
    "env": ["env_vars"],
    "docker": ["docker_state"],
    "process": ["process_state"],
    "tests": ["test_results"],
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_all(
    repo_path: Path,
    only: list[str] | None = None,
    output_path: Path | None = None,
    merge_with: Path | None = None,
) -> list[dict]:
    """
    Run all (or a subset of) collectors and return the context chunks.

    Parameters
    ----------
    repo_path:
        Absolute path to the repository root.
    only:
        List of collector names to run (e.g. ["deps", "git"]).
        None or empty -> run all.
    output_path:
        If given, write JSONL to this path.
    merge_with:
        If given, load existing JSONL and insert context chunks after the
        repo_overview (if present) and before all other chunks.

    Returns
    -------
    list[dict]
        All context chunks collected.
    """
    repo = Path(repo_path).resolve()
    active = only if only else list(_COLLECTORS.keys())

    context_chunks: list[dict] = []
    summary_lines: list[str] = []

    for name in active:
        fn = _COLLECTORS.get(name)
        if fn is None:
            print(f"  [warn] unknown collector: {name}", file=sys.stderr)
            continue
        try:
            chunks = fn(repo)
        except Exception as exc:
            print(f"  [warn] collector '{name}' failed: {exc}", file=sys.stderr)
            chunks = []
        context_chunks.extend(chunks)

        # Build summary line
        count = len(chunks)
        # human label: collect metadata hints
        details = ""
        if name == "deps" and chunks:
            files = [c["metadata"].get("file", "") for c in chunks if "file" in c.get("metadata", {})]
            if files:
                details = f"({', '.join(files[:3])})"
        elif name == "git" and chunks:
            git_h = next((c for c in chunks if c["type"] == "git_history"), None)
            if git_h:
                branch = git_h["metadata"].get("branch", "?")
                details = f"(branch: {branch})"
        elif name == "env" and chunks:
            m = chunks[0].get("metadata", {})
            details = f"({m.get('total_vars', '?')} vars, {m.get('redacted_count', '?')} redacted)"
        elif name == "docker" and chunks:
            m = chunks[0].get("metadata", {})
            n_running = len(m.get("running_containers", []))
            details = f"({n_running} containers running)"
        elif name == "process" and chunks:
            m = chunks[0].get("metadata", {})
            details = f"({m.get('process_count', '?')} relevant processes)"
        elif name == "tests" and chunks:
            m = chunks[0].get("metadata", {})
            passed = m.get("passed")
            fail = m.get("failures")
            cov = m.get("coverage_pct")
            parts = []
            if passed is not None:
                parts.append(f"{passed} passed")
            if fail is not None:
                parts.append(f"{fail} failed")
            if cov is not None:
                parts.append(f"{cov}% coverage")
            if parts:
                details = f"({', '.join(parts)})"

        pad = max(0, 7 - len(name))
        summary_lines.append(
            f"  {name}{' ' * pad}: {count} chunk{'s' if count != 1 else ''}  {details}"
        )

    # --merge-with: splice context chunks into existing JSONL
    if merge_with is not None and Path(merge_with).exists():
        existing: list[dict] = []
        try:
            for line in Path(merge_with).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
        except Exception as exc:
            print(f"  [warn] could not read merge-with file: {exc}", file=sys.stderr)
            existing = []

        # Find repo_overview position
        overview_idx = next(
            (i for i, c in enumerate(existing) if c.get("type") == "repo_overview"),
            -1,
        )
        insert_at = overview_idx + 1 if overview_idx >= 0 else 0
        merged = existing[:insert_at] + context_chunks + existing[insert_at:]
    else:
        merged = context_chunks

    if output_path is not None:
        out = Path(output_path)
        with out.open("w", encoding="utf-8") as fh:
            for chunk in merged:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Print summary
    print(f"Context collected from: {repo}")
    for line in summary_lines:
        print(line)
    print(f"Total: {len(context_chunks)} context chunks")
    if output_path:
        print(f"Written to: {output_path}")

    return context_chunks


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


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    repo = _pop_arg(args, "--repo") or "."
    output = _pop_arg(args, "--output") or "context_chunks.jsonl"
    only_raw = _pop_arg(args, "--only")
    merge_with = _pop_arg(args, "--merge-with")

    only = [s.strip() for s in only_raw.split(",") if s.strip()] if only_raw else None

    collect_all(
        repo_path=Path(repo),
        only=only,
        output_path=Path(output),
        merge_with=Path(merge_with) if merge_with else None,
    )


if __name__ == "__main__":
    main()
