#!/usr/bin/env python3
"""Profile the indexer to find performance bottlenecks."""
import cProfile
import pstats
import time
from pathlib import Path

from semantic_search_mcp.config import load_config
from semantic_search_mcp.database import Database
from semantic_search_mcp.embedder import Embedder
from semantic_search_mcp.indexer import FileIndexer


def profile_indexing(root_dir: Path, max_files: int = 20):
    """Profile indexing a directory."""
    config = load_config()

    # Use temp database
    db_path = Path("/tmp/profile_test.db")
    db_path.unlink(missing_ok=True)

    db = Database(db_path, embedding_dim=config.embedding_dim)
    embedder = Embedder(
        model_name=config.embedding_model,
        embedding_dim=config.embedding_dim,
        batch_size=config.embedding_batch_size,
        threads=config.embedding_threads,
        use_quantized=config.use_quantized,
    )
    indexer = FileIndexer(
        db, embedder, root_dir,
        chunk_overlap=config.chunk_overlap_tokens,
        max_chunk_tokens=config.max_chunk_tokens,
        max_file_size_kb=config.max_file_size_kb,
    )

    # Get files to index
    files = [f for f in root_dir.rglob("*") if indexer.gitignore.should_index(f)][:max_files]
    print(f"Profiling {len(files)} files...")

    # Time individual components
    timings = {
        "model_load": 0,
        "chunking": 0,
        "embedding": 0,
        "db_write": 0,
    }

    # Load model first
    print("Loading model...")
    start = time.time()
    _ = embedder.model
    timings["model_load"] = time.time() - start
    print(f"  Model load: {timings['model_load']:.2f}s")

    # Process each file
    for i, filepath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {filepath.name}")

        # Read and chunk
        start = time.time()
        chunks = indexer.chunker.chunk_file(filepath)
        chunk_time = time.time() - start
        timings["chunking"] += chunk_time
        print(f"  Chunking: {chunk_time:.3f}s ({len(chunks)} chunks)")

        if not chunks:
            continue

        # Embed
        texts = [c.content for c in chunks]
        start = time.time()
        embeddings = embedder.embed(texts)
        embed_time = time.time() - start
        timings["embedding"] += embed_time
        print(f"  Embedding: {embed_time:.3f}s ({len(texts)} texts)")

        # Store
        start = time.time()
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        content_hash = indexer._hash_content(content)
        file_id = db.upsert_file(str(filepath), content_hash, chunks[0].language)
        db.delete_chunks_for_file(file_id)
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = db.insert_chunk(
                file_id=file_id,
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                name=chunk.name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            db.insert_embedding(
                chunk_id=chunk_id,
                embedding=embedding,
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                file_path=str(filepath),
                name=chunk.name,
                preview=chunk.content[:200],
            )
        db_time = time.time() - start
        timings["db_write"] += db_time
        print(f"  DB write: {db_time:.3f}s")

    # Summary
    print("\n" + "="*50)
    print("TIMING SUMMARY")
    print("="*50)
    total = sum(timings.values())
    for name, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (t / total * 100) if total > 0 else 0
        print(f"  {name:15} {t:8.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':15} {total:8.2f}s")

    # Cleanup
    db.close()
    db_path.unlink(missing_ok=True)


def run_with_cprofile(root_dir: Path, max_files: int = 10):
    """Run with cProfile for detailed breakdown."""
    profiler = cProfile.Profile()
    profiler.enable()

    profile_indexing(root_dir, max_files)

    profiler.disable()

    print("\n" + "="*50)
    print("CPROFILE TOP 30 (by cumulative time)")
    print("="*50)
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print(f"Profiling indexer on: {root}")
    print(f"Max files: {max_files}")
    print()

    # Run simple timing first
    profile_indexing(root, max_files)

    # Uncomment for detailed cProfile:
    # run_with_cprofile(root, max_files)
