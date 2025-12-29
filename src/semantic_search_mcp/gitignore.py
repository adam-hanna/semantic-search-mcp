"""Gitignore-aware file filtering using pathspec."""
from pathlib import Path
from typing import Optional

import pathspec


# Always ignore these directories regardless of .gitignore
ALWAYS_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "target",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "coverage",
    ".coverage",
    "htmlcov",
    ".tox",
    ".nox",
    "*.egg-info",
}

# Code file extensions to index
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".lua",
    ".r",
    ".R",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".vim",
    ".el",
}


class GitignoreFilter:
    """Filter files based on .gitignore patterns and code extensions."""

    def __init__(self, root_dir: Path):
        """Initialize with root directory.

        Args:
            root_dir: Root directory to scan for .gitignore files
        """
        self.root = Path(root_dir).resolve()
        self.specs: dict[Path, pathspec.GitIgnoreSpec] = {}
        self._load_all_gitignores()

    def _load_all_gitignores(self):
        """Load all .gitignore files in the directory tree."""
        for gitignore in self.root.rglob(".gitignore"):
            try:
                with open(gitignore, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                self.specs[gitignore.parent] = pathspec.GitIgnoreSpec.from_lines(lines)
            except Exception:
                continue

    def _is_always_ignored(self, path: Path) -> bool:
        """Check if path is in an always-ignored directory."""
        for part in path.parts:
            if part in ALWAYS_IGNORE_DIRS:
                return True
            # Handle patterns like *.egg-info
            for pattern in ALWAYS_IGNORE_DIRS:
                if "*" in pattern:
                    import fnmatch
                    if fnmatch.fnmatch(part, pattern):
                        return True
        return False

    def _is_code_file(self, path: Path) -> bool:
        """Check if file has a code extension."""
        return path.suffix.lower() in CODE_EXTENSIONS

    def _matches_gitignore(self, path: Path) -> bool:
        """Check if path matches any gitignore pattern."""
        for gitignore_dir, spec in self.specs.items():
            try:
                # Get relative path from gitignore location
                rel_path = path.relative_to(gitignore_dir)
                if spec.match_file(str(rel_path)):
                    return True
            except ValueError:
                # Path is not relative to this gitignore
                continue
        return False

    def should_index(self, filepath: Path) -> bool:
        """Determine if a file should be indexed.

        Args:
            filepath: Path to check

        Returns:
            True if file should be indexed
        """
        path = Path(filepath).resolve()

        # Must be a file
        if not path.is_file():
            return False

        # Must have code extension
        if not self._is_code_file(path):
            return False

        # Check always-ignored directories
        if self._is_always_ignored(path):
            return False

        # Check gitignore patterns
        if self._matches_gitignore(path):
            return False

        return True

    def get_indexable_files(self) -> list[Path]:
        """Get all indexable files in the root directory.

        Returns:
            List of file paths that should be indexed
        """
        files = []
        for path in self.root.rglob("*"):
            if self.should_index(path):
                files.append(path)
        return files
