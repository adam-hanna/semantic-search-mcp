"""Tree-sitter based code chunking with overlap."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tree_sitter_languages


@dataclass
class Chunk:
    """A chunk of code extracted from a file."""
    content: str
    chunk_type: str  # 'function', 'class', 'method', 'module'
    name: Optional[str]
    start_line: int
    end_line: int
    language: str


# Map file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".xml": "xml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".elm": "elm",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".pl": "perl",
    ".pm": "perl",
}

# Node types to extract per language
EXTRACTABLE_NODES = {
    "python": ["function_definition", "class_definition", "async_function_definition"],
    "javascript": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
    "tsx": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
    "java": ["method_declaration", "class_declaration", "interface_declaration"],
    "go": ["function_declaration", "method_declaration", "type_declaration"],
    "rust": ["function_item", "impl_item", "struct_item", "enum_item"],
    "c": ["function_definition", "struct_specifier"],
    "cpp": ["function_definition", "class_specifier", "struct_specifier"],
    "c_sharp": ["method_declaration", "class_declaration", "interface_declaration"],
    "ruby": ["method", "class", "module"],
    "php": ["function_definition", "class_declaration", "method_declaration"],
    "kotlin": ["function_declaration", "class_declaration"],
    "scala": ["function_definition", "class_definition", "object_definition"],
    "swift": ["function_declaration", "class_declaration", "struct_declaration"],
}


class CodeChunker:
    """Extract semantic code chunks using tree-sitter."""

    def __init__(self, overlap_tokens: int = 50, max_tokens: int = 2000):
        """Initialize chunker.

        Args:
            overlap_tokens: Number of tokens to overlap between chunks
            max_tokens: Maximum tokens per chunk
        """
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens
        self._parsers: dict = {}

    def _get_parser(self, language: str):
        """Get or create a parser for the given language."""
        if language not in self._parsers:
            try:
                self._parsers[language] = tree_sitter_languages.get_parser(language)
            except Exception:
                return None
        return self._parsers[language]

    def _detect_language(self, filepath: Path) -> Optional[str]:
        """Detect language from file extension."""
        return EXTENSION_TO_LANGUAGE.get(filepath.suffix.lower())

    def _get_node_name(self, node, source_bytes: bytes, language: str) -> Optional[str]:
        """Extract the name from a function/class node."""
        # Look for identifier or name child
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

        # For some languages, name is nested deeper
        for child in node.children:
            if child.type in ("declarator", "function_declarator"):
                return self._get_node_name(child, source_bytes, language)

        return None

    def _node_to_chunk_type(self, node_type: str) -> str:
        """Convert tree-sitter node type to our chunk type."""
        if "class" in node_type or "struct" in node_type or "interface" in node_type:
            return "class"
        if "method" in node_type:
            return "method"
        if "function" in node_type or "arrow" in node_type:
            return "function"
        if "impl" in node_type or "enum" in node_type or "type" in node_type:
            return "class"
        return "block"

    def _add_overlap(self, content: str, lines: list[str], start_line: int, end_line: int) -> tuple[str, int, int]:
        """Add overlap context before the chunk."""
        # Approximate tokens as words (rough estimate)
        words_per_line = 10  # Rough average
        overlap_lines = max(1, self.overlap_tokens // words_per_line)

        # Add lines before
        actual_start = max(0, start_line - 1 - overlap_lines)
        prefix_lines = lines[actual_start:start_line - 1]

        if prefix_lines:
            content = "\n".join(prefix_lines) + "\n" + content
            start_line = actual_start + 1

        return content, start_line, end_line

    def chunk_file(self, filepath: Path) -> list[Chunk]:
        """Extract semantic chunks from a source file.

        Args:
            filepath: Path to the source file

        Returns:
            List of Chunk objects
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return []

        language = self._detect_language(filepath)
        if not language:
            return []

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        if not content.strip():
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        source_bytes = content.encode("utf-8")
        lines = content.split("\n")

        try:
            tree = parser.parse(source_bytes)
        except Exception:
            return []

        chunks = []
        extractable = EXTRACTABLE_NODES.get(language, [])

        def visit(node):
            """Recursively visit nodes and extract chunks."""
            if node.type in extractable:
                chunk_content = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Add overlap
                chunk_content, start_line, end_line = self._add_overlap(
                    chunk_content, lines, start_line, end_line
                )

                name = self._get_node_name(node, source_bytes, language)
                chunk_type = self._node_to_chunk_type(node.type)

                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=chunk_type,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                ))

            # Visit children
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        # If no chunks extracted, treat whole file as one chunk
        if not chunks and len(content) > 50:
            chunks.append(Chunk(
                content=content[:self.max_tokens * 4],  # Rough char limit
                chunk_type="module",
                name=filepath.stem,
                start_line=1,
                end_line=len(lines),
                language=language,
            ))

        return chunks
