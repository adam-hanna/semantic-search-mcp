# tests/test_server.py
"""Tests for MCP server module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semantic_search_mcp.server import create_server


@pytest.fixture
def mock_components(temp_dir):
    """Create mock components for server testing."""
    with patch("semantic_search_mcp.server.Database") as MockDB, \
         patch("semantic_search_mcp.server.Embedder") as MockEmbed, \
         patch("semantic_search_mcp.server.FileIndexer") as MockIndexer, \
         patch("semantic_search_mcp.server.HybridSearcher") as MockSearcher:

        mock_db = MagicMock()
        mock_db.get_stats.return_value = {"files": 10, "chunks": 50}
        MockDB.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.is_loaded.return_value = False
        MockEmbed.return_value = mock_embedder

        mock_indexer = MagicMock()
        mock_indexer.index_directory.return_value = {"files_indexed": 5, "total_chunks": 25}
        MockIndexer.return_value = mock_indexer

        mock_searcher = MagicMock()
        MockSearcher.return_value = mock_searcher

        yield {
            "db": mock_db,
            "embedder": mock_embedder,
            "indexer": mock_indexer,
            "searcher": mock_searcher,
            "temp_dir": temp_dir,
        }


def test_server_creates_mcp_instance(mock_components):
    """Server should create an MCP instance."""
    mcp = create_server(mock_components["temp_dir"])

    assert mcp is not None
    assert mcp.name == "SemanticCodeSearch"


def test_server_has_initialize_tool(mock_components):
    """Server should have an initialize tool."""
    mcp = create_server(mock_components["temp_dir"])

    # FastMCP registers tools on the internal _tool_manager
    tool_names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "initialize" in tool_names


def test_server_has_search_tool(mock_components):
    """Server should have a search_code tool."""
    mcp = create_server(mock_components["temp_dir"])

    tool_names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "search_code" in tool_names


def test_server_has_reindex_tool(mock_components):
    """Server should have a reindex_file tool."""
    mcp = create_server(mock_components["temp_dir"])

    tool_names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "reindex_file" in tool_names


def test_server_has_status_resource(mock_components):
    """Server should have a status resource."""
    mcp = create_server(mock_components["temp_dir"])

    # Check resources
    resource_uris = [r.uri for r in mcp._resource_manager._resources.values()]
    assert any("status" in str(uri) for uri in resource_uris)
