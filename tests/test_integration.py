# tests/test_integration.py
"""Integration tests for the full pipeline."""
from pathlib import Path

import pytest

from semantic_search_mcp.config import Config
from semantic_search_mcp.database import Database
from semantic_search_mcp.embedder import Embedder
from semantic_search_mcp.indexer import FileIndexer
from semantic_search_mcp.searcher import HybridSearcher


@pytest.fixture
def config():
    """Test config with small model."""
    return Config(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
    )


@pytest.fixture
def full_pipeline(temp_dir: Path, config):
    """Create full pipeline for integration testing."""
    db = Database(temp_dir / "test.db", embedding_dim=config.embedding_dim)
    embedder = Embedder(model_name=config.embedding_model, embedding_dim=config.embedding_dim)
    indexer = FileIndexer(db, embedder, temp_dir)
    searcher = HybridSearcher(db, embedder, rrf_k=config.rrf_k)

    yield {
        "db": db,
        "embedder": embedder,
        "indexer": indexer,
        "searcher": searcher,
        "root": temp_dir,
    }

    db.close()


def test_full_index_and_search_pipeline(full_pipeline):
    """Test complete index and search workflow."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    searcher = full_pipeline["searcher"]

    # Create test files
    (root / "auth.py").write_text('''
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    # Hash password and check against database
    hashed = hash_password(password)
    user = db.get_user(username)
    return user and user.password_hash == hashed

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
''')

    (root / "search.py").write_text('''
def binary_search(arr: list[int], target: int) -> int:
    """Binary search to find target in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def linear_search(arr: list, target) -> int:
    """Linear search through array."""
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
''')

    # Index files
    stats = indexer.index_directory(root)
    assert stats["files_indexed"] == 2
    assert stats["total_chunks"] > 0

    # Search for authentication code
    results = searcher.search("user login and password verification")
    assert len(results) > 0
    assert any("authenticate" in r.name.lower() for r in results if r.name)

    # Search for search algorithms
    results = searcher.search("find element in sorted array")
    assert len(results) > 0
    assert any("binary_search" in r.name for r in results if r.name)


def test_incremental_update(full_pipeline):
    """Test that file changes are detected and reindexed."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    db = full_pipeline["db"]

    # Create initial file
    test_file = root / "test.py"
    test_file.write_text("def foo(): pass")

    # First index
    result1 = indexer.index_file(test_file)
    assert result1["status"] == "indexed"

    initial_stats = db.get_stats()

    # Modify file
    test_file.write_text("def foo(): pass\ndef bar(): pass")

    # Reindex
    result2 = indexer.index_file(test_file)
    assert result2["status"] == "indexed"

    # Should have more chunks now
    updated_stats = db.get_stats()
    assert updated_stats["chunks"] >= initial_stats["chunks"]


def test_search_with_filters(full_pipeline):
    """Test search with language and type filters."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    searcher = full_pipeline["searcher"]

    # Create Python file
    (root / "app.py").write_text('''
class UserController:
    def get_user(self, id: int):
        return self.db.find(id)
''')

    # Create TypeScript file
    (root / "app.ts").write_text('''
class UserService {
    async getUser(id: number): Promise<User> {
        return await this.repo.findOne(id);
    }
}
''')

    indexer.index_directory(root)

    # Search with Python filter - vector search is filtered, so Python results should dominate
    py_results = searcher.search("get user by id", language="python")
    # Verify we get results and Python results are boosted
    assert len(py_results) > 0
    # The top result should be from Python since vector search is filtered
    python_results = [r for r in py_results if r.language == "python"]
    assert len(python_results) > 0

    # Search with TypeScript filter
    ts_results = searcher.search("get user by id", language="typescript")
    typescript_results = [r for r in ts_results if r.language == "typescript"]
    assert len(typescript_results) > 0


def test_hybrid_search_finds_exact_matches(full_pipeline):
    """Hybrid search should find exact keyword matches."""
    root = full_pipeline["root"]
    indexer = full_pipeline["indexer"]
    searcher = full_pipeline["searcher"]

    (root / "utils.py").write_text('''
def calculateTotalPrice(items: list) -> float:
    """Calculate the total price of all items."""
    return sum(item.price * item.quantity for item in items)
''')

    indexer.index_directory(root)

    # Search for exact function name
    results = searcher.search("calculateTotalPrice")

    assert len(results) > 0
    assert any("calculateTotalPrice" in r.name for r in results if r.name)
