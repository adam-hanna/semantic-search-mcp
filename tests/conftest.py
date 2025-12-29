"""Shared test fixtures."""
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    code = '''
def binary_search(arr: list[int], target: int) -> int:
    """Find target in sorted array using binary search."""
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


class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        """Fetch user by ID."""
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
'''
    file_path = temp_dir / "sample.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_typescript_file(temp_dir: Path) -> Path:
    """Create a sample TypeScript file for testing."""
    code = '''
interface User {
    id: number;
    name: string;
    email: string;
}

async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch user: ${response.status}`);
    }
    return response.json();
}

export class AuthService {
    private token: string | null = null;

    async login(username: string, password: string): Promise<boolean> {
        // Authentication logic here
        return true;
    }
}
'''
    file_path = temp_dir / "sample.ts"
    file_path.write_text(code)
    return file_path
