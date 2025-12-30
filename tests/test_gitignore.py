"""Tests for gitignore filtering module."""
from pathlib import Path

import pytest

from semantic_search_mcp.gitignore import GitignoreFilter


@pytest.fixture
def project_with_gitignore(temp_dir: Path) -> Path:
    """Create a project structure with .gitignore."""
    # Create .gitignore
    (temp_dir / ".gitignore").write_text("""
# Dependencies
node_modules/
__pycache__/
*.pyc

# Build output
dist/
build/

# IDE
.idea/
.vscode/

# Custom
secret.py
""")

    # Create some files
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "main.py").write_text("print('hello')")
    (temp_dir / "node_modules").mkdir()
    (temp_dir / "node_modules" / "package.json").write_text("{}")
    (temp_dir / "__pycache__").mkdir()
    (temp_dir / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"bytecode")
    (temp_dir / "secret.py").write_text("API_KEY = 'secret'")

    return temp_dir


def test_filter_allows_source_files(project_with_gitignore: Path):
    """Filter should allow regular source files."""
    filter = GitignoreFilter(project_with_gitignore)

    assert filter.should_index(project_with_gitignore / "src" / "main.py")


def test_filter_ignores_node_modules(project_with_gitignore: Path):
    """Filter should ignore node_modules."""
    filter = GitignoreFilter(project_with_gitignore)

    assert not filter.should_index(project_with_gitignore / "node_modules" / "package.json")


def test_filter_ignores_pycache(project_with_gitignore: Path):
    """Filter should ignore __pycache__."""
    filter = GitignoreFilter(project_with_gitignore)

    assert not filter.should_index(project_with_gitignore / "__pycache__" / "main.cpython-311.pyc")


def test_filter_ignores_gitignored_files(project_with_gitignore: Path):
    """Filter should ignore files matching .gitignore patterns."""
    filter = GitignoreFilter(project_with_gitignore)

    assert not filter.should_index(project_with_gitignore / "secret.py")


def test_filter_ignores_non_code_extensions(temp_dir: Path):
    """Filter should ignore non-code file extensions."""
    filter = GitignoreFilter(temp_dir)

    (temp_dir / "readme.md").write_text("# README")
    (temp_dir / "data.json").write_text("{}")
    (temp_dir / "image.png").write_bytes(b"PNG")

    assert not filter.should_index(temp_dir / "readme.md")
    assert not filter.should_index(temp_dir / "data.json")
    assert not filter.should_index(temp_dir / "image.png")


def test_filter_allows_code_extensions(temp_dir: Path):
    """Filter should allow code file extensions."""
    filter = GitignoreFilter(temp_dir)

    for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]:
        file = temp_dir / f"code{ext}"
        file.write_text("code")
        assert filter.should_index(file), f"Should index {ext} files"


def test_filter_always_ignores_git_directory(temp_dir: Path):
    """Filter should always ignore .git directory."""
    filter = GitignoreFilter(temp_dir)

    (temp_dir / ".git").mkdir()
    (temp_dir / ".git" / "config").write_text("[core]")

    assert not filter.should_index(temp_dir / ".git" / "config")


def test_filter_handles_nested_gitignore(temp_dir: Path):
    """Filter should respect nested .gitignore files."""
    # Root gitignore
    (temp_dir / ".gitignore").write_text("*.log")

    # Nested gitignore
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / ".gitignore").write_text("temp/")
    (temp_dir / "src" / "temp").mkdir()
    (temp_dir / "src" / "temp" / "cache.py").write_text("")
    (temp_dir / "src" / "main.py").write_text("")

    filter = GitignoreFilter(temp_dir)

    assert filter.should_index(temp_dir / "src" / "main.py")
    assert not filter.should_index(temp_dir / "src" / "temp" / "cache.py")


def test_runtime_exclusions(tmp_path):
    """Runtime exclusions should be respected by should_index."""
    # Create a Python file
    py_file = tmp_path / "src" / "main.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("print('hello')")

    gf = GitignoreFilter(tmp_path)

    # File should be indexable initially
    assert gf.should_index(py_file) is True

    # Add runtime exclusion
    gf.add_exclusions(["src"])
    assert gf.should_index(py_file) is False
    assert "src" in gf.get_exclusions()

    # Remove exclusion
    gf.remove_exclusions(["src"])
    assert gf.should_index(py_file) is True
    assert "src" not in gf.get_exclusions()


def test_runtime_exclusions_glob_patterns(tmp_path):
    """Runtime exclusions should support glob patterns."""
    test_file = tmp_path / "test_main.py"
    test_file.write_text("def test(): pass")

    src_file = tmp_path / "main.py"
    src_file.write_text("def main(): pass")

    gf = GitignoreFilter(tmp_path)

    # Both should be indexable initially
    assert gf.should_index(test_file) is True
    assert gf.should_index(src_file) is True

    # Exclude test files
    gf.add_exclusions(["test_*.py"])
    assert gf.should_index(test_file) is False
    assert gf.should_index(src_file) is True
