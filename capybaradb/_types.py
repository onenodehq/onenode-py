from typing import TypedDict

"""
Type definitions for CapybaraDB API responses.

This module defines TypedDict classes that represent the structure of responses
from the CapybaraDB API, particularly for semantic search queries.
"""

class QueryMatch(TypedDict):
    """
    Represents a single match from a semantic search query.
    
    Each match contains:
    - chunk: The text chunk that matched the query
    - path: The document field path where the match was found
    - chunk_n: The index of the chunk within the field
    - score: The similarity score (0-1) indicating how well the chunk matches the query
    - document: The full document containing the match
    
    Example:
        ```python
        {
            "chunk": "This is the matching text fragment...",
            "path": "content",
            "chunk_n": 0,
            "score": 0.89,
            "document": {"_id": "...", "title": "Document Title", ...}
        }
        ```
    """
    chunk: str  # The text chunk that matched the query
    path: str   # The document field path where the match was found
    chunk_n: int  # The index of the chunk within the field
    score: float  # The similarity score (0-1)
    document: dict  # The full document containing the match


class QueryResponse(TypedDict):
    """
    Represents the complete response from a semantic search query.
    
    Contains a list of matches, sorted by relevance (highest score first).
    
    Example:
        ```python
        {
            "matches": [
                {
                    "chunk": "First matching text...",
                    "path": "content",
                    "chunk_n": 0,
                    "score": 0.92,
                    "document": {"_id": "...", ...}
                },
                {
                    "chunk": "Second matching text...",
                    "path": "description",
                    "chunk_n": 1,
                    "score": 0.85,
                    "document": {"_id": "...", ...}
                }
            ]
        }
        ```
    """
    matches: list[QueryMatch]  # List of matches, sorted by relevance
