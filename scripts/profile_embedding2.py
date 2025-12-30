#!/usr/bin/env python3
"""Profile embedding with realistic code chunks."""
import time
from pathlib import Path
from semantic_search_mcp.embedder import Embedder
from semantic_search_mcp.chunker import CodeChunker

def get_real_chunks():
    """Get real code chunks from this codebase."""
    chunker = CodeChunker()
    chunks = []

    for f in Path("src/semantic_search_mcp").glob("*.py"):
        file_chunks = chunker.chunk_file(f)
        chunks.extend([c.content for c in file_chunks])

    return chunks


def test_with_real_chunks():
    """Test embedding with real code chunks."""
    chunks = get_real_chunks()
    print(f"Got {len(chunks)} real code chunks")
    print(f"Avg chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
    print(f"Max chunk size: {max(len(c) for c in chunks)} chars")
    print()

    embedder = Embedder()

    print("Loading model...")
    start = time.time()
    _ = embedder.model
    print(f"Model load: {time.time() - start:.2f}s\n")

    # Test batch sizes
    for batch_size in [1, 4, 8, 16, 32, 64, len(chunks)]:
        embedder.batch_size = batch_size

        start = time.time()
        results = embedder.embed(chunks)
        elapsed = time.time() - start

        per_text = elapsed / len(chunks) * 1000
        print(f"batch_size={batch_size:3d}: {elapsed:.2f}s total, {per_text:.0f}ms/chunk")


if __name__ == "__main__":
    test_with_real_chunks()
