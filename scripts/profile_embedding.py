#!/usr/bin/env python3
"""Profile embedding specifically."""
import time
from semantic_search_mcp.embedder import Embedder

# Test texts of varying sizes
TEST_TEXTS = [
    "def hello(): pass",
    "def foo(x): return x * 2",
    """def complex_function(a, b, c):
    result = []
    for i in range(a):
        for j in range(b):
            result.append(i * j + c)
    return result
""",
] * 5  # 15 texts total

def test_batch_sizes():
    """Test different batch sizes."""
    embedder = Embedder()

    print("Loading model...")
    start = time.time()
    _ = embedder.model
    print(f"Model load: {time.time() - start:.2f}s\n")

    for batch_size in [1, 2, 4, 8, 16, 32, len(TEST_TEXTS)]:
        embedder.batch_size = batch_size

        start = time.time()
        results = embedder.embed(TEST_TEXTS)
        elapsed = time.time() - start

        per_text = elapsed / len(TEST_TEXTS) * 1000
        print(f"batch_size={batch_size:2d}: {elapsed:.3f}s total, {per_text:.1f}ms/text ({len(results)} embeddings)")


def test_direct_fastembed():
    """Test fastembed directly without our wrapper."""
    from fastembed import TextEmbedding

    print("\nDirect fastembed test:")
    model = TextEmbedding("jinaai/jina-embeddings-v2-base-code")

    # Single call with all texts
    start = time.time()
    results = list(model.embed(TEST_TEXTS))
    elapsed = time.time() - start
    per_text = elapsed / len(TEST_TEXTS) * 1000
    print(f"All at once: {elapsed:.3f}s total, {per_text:.1f}ms/text")

    # One at a time
    start = time.time()
    for text in TEST_TEXTS:
        list(model.embed([text]))
    elapsed = time.time() - start
    per_text = elapsed / len(TEST_TEXTS) * 1000
    print(f"One at a time: {elapsed:.3f}s total, {per_text:.1f}ms/text")


if __name__ == "__main__":
    test_batch_sizes()
    test_direct_fastembed()
