from typing import TypedDict

"""Type definitions for OneNode API responses."""

class QueryMatch(TypedDict):
    """Single match from a semantic search query."""
    chunk: str  # Text chunk that matched the query
    path: str   # Document field path
    chunk_n: int  # Index of the chunk
    score: float  # Similarity score (0-1)
    document: dict  # Full document containing the match


class QueryResponse(TypedDict):
    """Complete response from a semantic search query."""
    matches: list[QueryMatch]  # Matches sorted by relevance
