"""
tests/test_multilang_parser.py — Unit tests for the multi-language parser suite.

All fixtures are inline strings (no files on disk).
Uses only stdlib: unittest, pathlib, tempfile, sys.

Run:
    python -m pytest tests/test_multilang_parser.py -v
    # or
    python -m unittest tests.test_multilang_parser -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path so parsers/ and parse_repo.py are importable
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helper: write an inline fixture to a temp dir and return (path, root)
# ---------------------------------------------------------------------------

def _write_temp(suffix: str, content: str):
    """Create a named temp file with given content; return (Path, root_dir_Path)."""
    tmp_dir = tempfile.mkdtemp()
    root = Path(tmp_dir)
    fname = f"fixture{suffix}"
    fpath = root / fname
    fpath.write_text(content, encoding="utf-8")
    return fpath, root


# ===========================================================================
# TypeScript parser tests
# ===========================================================================

class TestTypeScriptParser(unittest.TestCase):

    def _parser(self):
        from parsers.typescript_parser import parse_file, can_parse
        return parse_file, can_parse

    def test_can_parse_ts_extension(self):
        _, can_parse = self._parser()
        self.assertTrue(can_parse(Path("app/src/auth.ts")))
        self.assertTrue(can_parse(Path("utils.tsx")))
        self.assertFalse(can_parse(Path("main.py")))
        self.assertFalse(can_parse(Path("schema.sql")))

    def test_typescript_function_detection(self):
        """Three different function styles should each produce a function chunk."""
        code = """\
function greet(name: string): string {
    return `Hello, ${name}!`;
}

const farewell = (name: string) => {
    return `Goodbye, ${name}`;
};

const asyncFetch = async (url: string) => {
    const res = await fetch(url);
    return res.json();
};
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".ts", code)
        chunks = parse_file(fpath, root)

        func_names = [c["metadata"].get("name") for c in chunks if c["type"] == "function"]
        self.assertIn("greet", func_names, "function declaration not detected")
        self.assertIn("farewell", func_names, "arrow function not detected")
        self.assertIn("asyncFetch", func_names, "async arrow function not detected")

    def test_typescript_interface_detection(self):
        """Interface declarations should produce chunks of type 'interface'."""
        code = """\
export interface User {
    id: number;
    name: string;
    email?: string;
}

interface ApiResponse<T> {
    data: T;
    status: number;
}
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".ts", code)
        chunks = parse_file(fpath, root)

        iface_chunks = [c for c in chunks if c["type"] == "interface"]
        iface_names = [c["metadata"].get("name") for c in iface_chunks]
        self.assertIn("User", iface_names)
        self.assertIn("ApiResponse", iface_names)
        for c in iface_chunks:
            self.assertEqual(c["language"], "typescript")

    def test_typescript_class_detection(self):
        """Class definitions should produce class chunks with correct metadata."""
        code = """\
class Animal {
    constructor(public name: string) {}
    speak(): string { return `${this.name} speaks`; }
}

export class Dog extends Animal {
    bark() { return 'Woof!'; }
}
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".ts", code)
        chunks = parse_file(fpath, root)

        class_chunks = [c for c in chunks if c["type"] == "class"]
        class_names = [c["metadata"].get("name") for c in class_chunks]
        self.assertIn("Animal", class_names)
        self.assertIn("Dog", class_names)

        dog_chunk = next(c for c in class_chunks if c["metadata"].get("name") == "Dog")
        self.assertEqual(dog_chunk["metadata"].get("base"), "Animal")

    def test_typescript_file_chunk_contains_imports_and_exports(self):
        """The file overview chunk should list imports and exports."""
        code = """\
import { useState } from 'react';
import axios from 'axios';

export const fetchUser = async (id: number) => {
    const res = await axios.get(`/api/users/${id}`);
    return res.data;
};

export default function App() { return null; }
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".tsx", code)
        chunks = parse_file(fpath, root)

        file_chunk = next(c for c in chunks if c["type"] == "file")
        self.assertIn("react", file_chunk["metadata"]["imports"])
        self.assertIn("axios", file_chunk["metadata"]["imports"])
        self.assertTrue(file_chunk["metadata"]["has_default_export"])


# ===========================================================================
# Go parser tests
# ===========================================================================

class TestGoParser(unittest.TestCase):

    def _parser(self):
        from parsers.go_parser import parse_file, can_parse
        return parse_file, can_parse

    def test_can_parse_go_extension(self):
        _, can_parse = self._parser()
        self.assertTrue(can_parse(Path("main.go")))
        self.assertFalse(can_parse(Path("main.py")))

    def test_go_struct_detection(self):
        """Struct definitions should produce chunks of type 'struct'."""
        code = """\
package models

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

type Post struct {
    ID      int    `json:"id"`
    Title   string `json:"title"`
    Content string `json:"content"`
}
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".go", code)
        chunks = parse_file(fpath, root)

        struct_chunks = [c for c in chunks if c["type"] == "struct"]
        struct_names = [c["metadata"]["name"] for c in struct_chunks]
        self.assertIn("User", struct_names)
        self.assertIn("Post", struct_names)

        for c in struct_chunks:
            self.assertEqual(c["language"], "go")
            self.assertTrue(c["id"].startswith("struct::"))

    def test_go_func_with_receiver(self):
        """Method with receiver should include receiver info in text and metadata."""
        code = """\
package models

type User struct {
    Name string
}

func (u User) Greet() string {
    return "Hello, " + u.Name
}

func (u *User) SetName(name string) {
    u.Name = name
}
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".go", code)
        chunks = parse_file(fpath, root)

        method_chunks = [c for c in chunks if c["type"] == "function" and c["metadata"].get("receiver")]
        self.assertTrue(len(method_chunks) >= 2, "Expected at least 2 method chunks")

        greet = next((c for c in method_chunks if c["metadata"]["name"] == "Greet"), None)
        self.assertIsNotNone(greet)
        self.assertIn("User", greet["text"])  # receiver in text
        self.assertEqual(greet["metadata"]["receiver"], "User")

    def test_go_top_level_function_detection(self):
        """Top-level Go functions (no receiver) should be detected."""
        code = """\
package main

import "fmt"

func main() {
    fmt.Println("hello")
}

func add(a, b int) int {
    return a + b
}
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".go", code)
        chunks = parse_file(fpath, root)

        func_names = [c["metadata"]["name"] for c in chunks if c["type"] == "function"]
        self.assertIn("main", func_names)
        self.assertIn("add", func_names)

    def test_go_interface_detection(self):
        """Interface definitions should produce interface chunks."""
        code = """\
package api

type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
}
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".go", code)
        chunks = parse_file(fpath, root)

        iface_chunks = [c for c in chunks if c["type"] == "interface"]
        self.assertTrue(len(iface_chunks) >= 1)
        self.assertEqual(iface_chunks[0]["metadata"]["name"], "Handler")


# ===========================================================================
# YAML parser tests
# ===========================================================================

class TestYAMLParser(unittest.TestCase):

    def _parser(self):
        from parsers.yaml_parser import parse_file, can_parse
        return parse_file, can_parse

    def test_can_parse_yaml_extension(self):
        _, can_parse = self._parser()
        self.assertTrue(can_parse(Path("docker-compose.yml")))
        self.assertTrue(can_parse(Path("config.yaml")))
        self.assertFalse(can_parse(Path("main.go")))

    def test_yaml_docker_compose_services(self):
        """Docker Compose services should each produce a separate chunk."""
        yaml = """\
version: "3.8"
services:
  web:
    image: nginx:alpine
    ports:
      - "80:80"
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: mydb
  redis:
    image: redis:7
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".yml", yaml)
        chunks = parse_file(fpath, root)

        service_chunks = [c for c in chunks if c["type"] == "compose_service"]
        service_names = [c["metadata"]["service"] for c in service_chunks]
        self.assertIn("web", service_names)
        self.assertIn("db", service_names)
        self.assertIn("redis", service_names)
        self.assertEqual(len(service_chunks), 3)

        for c in service_chunks:
            self.assertEqual(c["language"], "yaml")

    def test_yaml_k8s_manifest(self):
        """Kubernetes manifests with --- separator should produce separate chunks."""
        yaml = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
---
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".yaml", yaml)
        chunks = parse_file(fpath, root)

        # Should have separate chunks for Deployment and Service
        self.assertGreaterEqual(len(chunks), 2)
        kinds = [c["metadata"].get("kind") for c in chunks]
        self.assertIn("Deployment", kinds)
        self.assertIn("Service", kinds)

    def test_yaml_generic_fallback(self):
        """Generic YAML (no services/jobs/kind) falls back to a single file chunk."""
        yaml = """\
database:
  host: localhost
  port: 5432
  name: mydb
logging:
  level: INFO
  format: json
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".yaml", yaml)
        chunks = parse_file(fpath, root)

        # Should produce at least one chunk
        self.assertGreaterEqual(len(chunks), 1)
        # Top-level keys should be detected
        file_chunk = chunks[0]
        top_keys = file_chunk["metadata"].get("top_level_keys", [])
        self.assertIn("database", top_keys)


# ===========================================================================
# Markdown parser tests
# ===========================================================================

class TestMarkdownParser(unittest.TestCase):

    def _parser(self):
        from parsers.markdown_parser import parse_file, can_parse
        return parse_file, can_parse

    def test_can_parse_md_extension(self):
        _, can_parse = self._parser()
        self.assertTrue(can_parse(Path("README.md")))
        self.assertTrue(can_parse(Path("docs/guide.mdx")))
        self.assertFalse(can_parse(Path("main.py")))

    def test_markdown_sections(self):
        """Three H2 sections should produce 3 separate chunks."""
        md = """\
## Installation

Run `pip install mypackage`.

## Usage

Import and call the main function.

## Configuration

Set the environment variables.
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".md", md)
        chunks = parse_file(fpath, root)

        self.assertEqual(len(chunks), 3)
        headings = [c["metadata"]["heading"] for c in chunks]
        self.assertIn("Installation", headings)
        self.assertIn("Usage", headings)
        self.assertIn("Configuration", headings)

        for c in chunks:
            self.assertEqual(c["type"], "doc")
            self.assertEqual(c["language"], "markdown")

    def test_markdown_h1_h2_h3_mixed(self):
        """H1, H2, H3 headings should all be detected."""
        md = """\
# Title

Intro.

## Section One

Content one.

### Subsection

Deep content.
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".md", md)
        chunks = parse_file(fpath, root)

        # 3 headings → 3 chunks: Title (H1), Section One (H2), Subsection (H3)
        self.assertEqual(len(chunks), 3)
        levels = {c["metadata"]["heading"]: c["metadata"]["level"] for c in chunks}
        self.assertEqual(levels["Title"], 1)
        self.assertEqual(levels["Section One"], 2)
        self.assertEqual(levels["Subsection"], 3)

    def test_markdown_no_headings_single_chunk(self):
        """Markdown without headings produces exactly one fallback chunk."""
        md = "Just some plain text without any headings.\n\nAnother paragraph.\n"
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".md", md)
        chunks = parse_file(fpath, root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "doc")

    def test_markdown_truncation(self):
        """Sections longer than 2000 chars should be truncated."""
        long_content = "word " * 1000  # 5000 chars
        md = f"## Long Section\n\n{long_content}\n"
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".md", md)
        chunks = parse_file(fpath, root)

        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]["text"]), 2100)  # 2000 + small overhead


# ===========================================================================
# SQL parser tests
# ===========================================================================

class TestSQLParser(unittest.TestCase):

    def _parser(self):
        from parsers.sql_parser import parse_file, can_parse
        return parse_file, can_parse

    def test_can_parse_sql_extension(self):
        _, can_parse = self._parser()
        self.assertTrue(can_parse(Path("schema.sql")))
        self.assertFalse(can_parse(Path("main.py")))

    def test_sql_create_table(self):
        """CREATE TABLE statements should be detected, name extracted."""
        sql = """\
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(200) NOT NULL,
    body TEXT
);
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".sql", sql)
        chunks = parse_file(fpath, root)

        sql_chunks = [c for c in chunks if c["type"] == "sql_object"]
        obj_names = [c["metadata"]["object_name"] for c in sql_chunks]
        self.assertIn("users", obj_names)
        self.assertIn("posts", obj_names)

        for c in sql_chunks:
            self.assertEqual(c["language"], "sql")
            self.assertTrue(c["id"].startswith("sql::"))

    def test_sql_create_view_and_index(self):
        """CREATE VIEW and CREATE INDEX are also detected."""
        sql = """\
CREATE VIEW active_users AS
    SELECT * FROM users WHERE active = TRUE;

CREATE INDEX idx_users_email ON users(email);
"""
        parse_file, _ = self._parser()
        fpath, root = _write_temp(".sql", sql)
        chunks = parse_file(fpath, root)

        sql_chunks = [c for c in chunks if c["type"] == "sql_object"]
        types = {c["metadata"]["object_type"] for c in sql_chunks}
        self.assertIn("VIEW", types)
        self.assertIn("INDEX", types)


# ===========================================================================
# Python parser wrap test
# ===========================================================================

class TestPythonParser(unittest.TestCase):

    def test_python_parser_wraps_existing(self):
        """python_parser should delegate to parse_repo and add language='python'."""
        from parsers.python_parser import parse_file, can_parse

        self.assertTrue(can_parse(Path("app.py")))
        self.assertFalse(can_parse(Path("app.ts")))

        code = """\
\"\"\"Simple module.\"\"\"

def hello(name: str) -> str:
    \"\"\"Return greeting.\"\"\"
    return f"Hello, {name}"

class Greeter:
    def greet(self) -> str:
        return "Hi!"
"""
        fpath, root = _write_temp(".py", code)
        chunks = parse_file(fpath, root)

        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c.get("language"), "python",
                             f"chunk {c['id']} missing language='python'")

        chunk_types = {c["type"] for c in chunks}
        self.assertTrue(
            chunk_types & {"file", "function", "class"},
            "Expected file/function/class chunks from python_parser"
        )

        func_chunks = [c for c in chunks if c["type"] == "function"]
        func_names = [c["metadata"].get("name") for c in func_chunks]
        self.assertIn("hello", func_names)


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistry(unittest.TestCase):

    def test_registry_can_parse(self):
        """Registry should route files to the correct parsers."""
        from parsers import find_parser

        ts_path = Path("src/app.ts")
        go_path = Path("main.go")
        py_path = Path("server.py")
        sql_path = Path("schema.sql")
        md_path = Path("README.md")
        yaml_path = Path("config.yml")
        unknown_path = Path("binary.exe")

        self.assertIsNotNone(find_parser(ts_path), "TypeScript file should have a parser")
        self.assertIsNotNone(find_parser(go_path), "Go file should have a parser")
        self.assertIsNotNone(find_parser(py_path), "Python file should have a parser")
        self.assertIsNotNone(find_parser(sql_path), "SQL file should have a parser")
        self.assertIsNotNone(find_parser(md_path), "Markdown file should have a parser")
        self.assertIsNotNone(find_parser(yaml_path), "YAML file should have a parser")
        self.assertIsNone(find_parser(unknown_path), ".exe should have no parser")

    def test_registry_language_list(self):
        """list_parser_languages should include all six expected languages."""
        from parsers import list_parser_languages
        langs = list_parser_languages()
        for expected in ["python", "typescript", "go", "yaml", "markdown", "sql"]:
            self.assertIn(expected, langs, f"Language '{expected}' missing from registry")


# ===========================================================================
# Windows path / forward slash test
# ===========================================================================

class TestWindowsPathForwardSlash(unittest.TestCase):

    def test_windows_path_forward_slash_typescript(self):
        """Chunk IDs must use forward slashes even when the path contains backslashes."""
        from parsers.typescript_parser import parse_file

        code = "export function hello() { return 'world'; }\n"

        # Simulate a Windows-style nested path
        import tempfile, os
        tmp_dir = tempfile.mkdtemp()
        root = Path(tmp_dir)
        subdir = root / "src" / "utils"
        subdir.mkdir(parents=True)
        fpath = subdir / "helpers.ts"
        fpath.write_text(code, encoding="utf-8")

        chunks = parse_file(fpath, root)

        for chunk in chunks:
            self.assertNotIn("\\", chunk["id"],
                             f"Backslash found in chunk ID: {chunk['id']!r}")
            # Should be src/utils/helpers.ts
            if chunk["type"] == "file":
                self.assertEqual(chunk["id"], "file::src/utils/helpers.ts")

    def test_windows_path_forward_slash_go(self):
        """Go parser chunk IDs must use forward slashes."""
        from parsers.go_parser import parse_file

        code = "package main\n\nfunc Hello() string { return \"hello\" }\n"

        import tempfile
        tmp_dir = tempfile.mkdtemp()
        root = Path(tmp_dir)
        subdir = root / "pkg" / "greet"
        subdir.mkdir(parents=True)
        fpath = subdir / "greet.go"
        fpath.write_text(code, encoding="utf-8")

        chunks = parse_file(fpath, root)
        for chunk in chunks:
            self.assertNotIn("\\", chunk["id"],
                             f"Backslash found in chunk ID: {chunk['id']!r}")


# ===========================================================================
# Graceful degradation / fallback test
# ===========================================================================

class TestGracefulFallback(unittest.TestCase):

    def test_parse_file_fallback_on_error(self):
        """Registry parse_file should return a fallback file chunk on parser error."""
        # We mock a bad .py file (valid extension but content that causes issues)
        import tempfile
        from parsers import parse_file as registry_parse_file

        tmp_dir = tempfile.mkdtemp()
        root = Path(tmp_dir)
        # Write a file with invalid Python syntax — the AST parser will raise SyntaxError
        bad_py = root / "broken.py"
        bad_py.write_text("def foo(\n  # unclosed paren — syntax error\n", encoding="utf-8")

        # parse_repo.chunks_for_file prints a warning and returns nothing on SyntaxError;
        # our python_parser wraps that and should still return something
        chunks = registry_parse_file(bad_py, root)
        # Either returns empty (parse_repo skips it) or a fallback — either is acceptable,
        # but we should NOT get an unhandled exception.
        # The important thing is that no exception propagates.
        self.assertIsInstance(chunks, list)

    def test_markdown_read_error_fallback(self):
        """markdown_parser on a non-existent file should return a single error chunk."""
        from parsers.markdown_parser import parse_file

        import tempfile
        tmp_dir = tempfile.mkdtemp()
        root = Path(tmp_dir)
        ghost = root / "ghost.md"
        # DO NOT create the file

        chunks = parse_file(ghost, root)
        self.assertEqual(len(chunks), 1)
        self.assertIn("read error", chunks[0]["text"].lower())

    def test_sql_no_create_statements_fallback(self):
        """SQL files with no CREATE statements should still produce a chunk."""
        from parsers.sql_parser import parse_file

        sql = "INSERT INTO users (name) VALUES ('Alice');\nSELECT * FROM users;\n"
        fpath, root = _write_temp(".sql", sql)
        chunks = parse_file(fpath, root)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["language"], "sql")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
