# src/semantic_search_mcp/searcher.py
"""Hybrid vector + FTS search with Reciprocal Rank Fusion."""
import fnmatch
from dataclasses import dataclass
from typing import Optional

from semantic_search_mcp.database import Database, serialize_embedding
from semantic_search_mcp.embedder import Embedder


@dataclass
class SearchResult:
    """A search result with relevance score."""
    chunk_id: int
    file_path: str
    name: Optional[str]
    chunk_type: str
    content: str
    preview: str
    start_line: int
    end_line: int
    language: str
    score: float  # Combined RRF score


class HybridSearcher:
    """Combine vector similarity and FTS for hybrid search."""

    def __init__(self, db: Database, embedder: Embedder, rrf_k: int = 60):
        """Initialize hybrid searcher.

        Args:
            db: Database instance
            embedder: Embedder instance
            rrf_k: Reciprocal Rank Fusion constant (default 60)
        """
        self.db = db
        self.embedder = embedder
        self.rrf_k = rrf_k

    def _vector_search(
        self,
        query_embedding: list[float],
        max_results: int,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """Perform vector similarity search.

        Returns:
            List of (chunk_id, distance) tuples
        """
        # sqlite-vec uses k=? as the limit, no separate LIMIT clause needed
        sql = """
            SELECT chunk_id, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
              AND k = ?
        """
        params: list = [serialize_embedding(query_embedding), max_results * 2]

        if language:
            sql += " AND language = ?"
            params.append(language)
        if chunk_type:
            sql += " AND chunk_type = ?"
            params.append(chunk_type)

        sql += " ORDER BY distance"

        rows = self.db.conn.execute(sql, params).fetchall()
        return [(row[0], row[1]) for row in rows]

    def _fts_search(
        self,
        query: str,
        max_results: int,
    ) -> list[tuple[int, float]]:
        """Perform FTS5 keyword search.

        Returns:
            List of (chunk_id, bm25_score) tuples
        """
        # Escape FTS5 special characters
        safe_query = query.replace('"', '""')

        sql = """
            SELECT rowid, bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """

        try:
            rows = self.db.conn.execute(sql, (safe_query, max_results * 2)).fetchall()
            return [(row[0], row[1]) for row in rows]
        except Exception:
            # FTS query failed, return empty
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[int, float]],
        fts_results: list[tuple[int, float]],
    ) -> dict[int, float]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list

        Args:
            vector_results: (chunk_id, distance) from vector search
            fts_results: (chunk_id, bm25_score) from FTS

        Returns:
            Dict mapping chunk_id to combined RRF score
        """
        scores: dict[int, float] = {}

        # Add vector search scores
        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)

        # Add FTS scores
        for rank, (chunk_id, _) in enumerate(fts_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)

        return scores

    def _get_chunk_details(self, chunk_id: int) -> Optional[dict]:
        """Get full chunk details from database."""
        row = self.db.conn.execute("""
            SELECT c.id, c.content, c.chunk_type, c.name, c.start_line, c.end_line,
                   f.path as file_path, f.language
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.id = ?
        """, (chunk_id,)).fetchone()

        if row:
            return {
                "id": row[0],
                "content": row[1],
                "chunk_type": row[2],
                "name": row[3],
                "start_line": row[4],
                "end_line": row[5],
                "file_path": row[6],
                "language": row[7],
            }
        return None

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        file_pattern: Optional[str] = None,
        recency_boost: bool = False,
    ) -> list[SearchResult]:
        """Perform hybrid search.

        Args:
            query: Natural language search query
            max_results: Maximum results to return
            min_score: Minimum RRF score threshold (0-1 normalized)
            language: Filter by programming language
            chunk_type: Filter by chunk type
            file_pattern: Glob pattern to filter files
            recency_boost: Boost recent files (not yet implemented)

        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Get results from both sources
        vector_results = self._vector_search(
            query_embedding, max_results * 2, language, chunk_type
        )
        fts_results = self._fts_search(query, max_results * 2)

        # Combine with RRF
        combined_scores = self._reciprocal_rank_fusion(vector_results, fts_results)

        # Normalize scores to 0-1 range
        if combined_scores:
            max_score = max(combined_scores.values())
            combined_scores = {k: v / max_score for k, v in combined_scores.items()}

        # Sort by score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Build results
        results = []
        for chunk_id in sorted_ids:
            if len(results) >= max_results:
                break

            score = combined_scores[chunk_id]
            if score < min_score:
                continue

            details = self._get_chunk_details(chunk_id)
            if not details:
                continue

            # Apply file pattern filter
            if file_pattern:
                if not fnmatch.fnmatch(details["file_path"], file_pattern):
                    continue

            results.append(SearchResult(
                chunk_id=chunk_id,
                file_path=details["file_path"],
                name=details["name"],
                chunk_type=details["chunk_type"],
                content=details["content"],
                preview=details["content"][:200],
                start_line=details["start_line"],
                end_line=details["end_line"],
                language=details["language"],
                score=score,
            ))

        return results
