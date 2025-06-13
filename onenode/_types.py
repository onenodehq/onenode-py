from typing import TypedDict, Optional, List, Any

"""Type definitions for OneNode API responses."""

class QueryMatch:
    """Single match from a semantic search query with attribute-style access.
    
    Provides dot notation access for fixed fields:
    - match.chunk - Text chunk that matched the query
    - match.path - Document field path  
    - match.chunk_n - Index of the chunk
    - match.score - Similarity score (0-1)
    - match.document - Full document containing the match (regular dict)
    - match.values - Embedding vector values (optional, when include_values=True)
    """
    
    def __init__(self, data: dict):
        """Initialize QueryMatch with raw response data."""
        self._data = data
    
    @property
    def chunk(self) -> str:
        """Text chunk that matched the query."""
        return self._data.get('chunk', '')
    
    @property
    def path(self) -> str:
        """Document field path where the match was found."""
        return self._data.get('path', '')
    
    @property
    def chunk_n(self) -> int:
        """Index of the chunk."""
        return self._data.get('chunk_n', 0)
    
    @property
    def score(self) -> float:
        """Similarity score (0-1)."""
        return self._data.get('score', 0.0)
    
    @property
    def document(self) -> dict:
        """Full document containing the match (regular dict)."""
        return self._data.get('document', {})
    
    @property
    def values(self) -> Optional[List[float]]:
        """Embedding vector values (only present when include_values=True)."""
        return self._data.get('values')
    
    def __getitem__(self, key):
        """Support bracket notation access for backward compatibility."""
        return self._data[key]
    
    def __contains__(self, key):
        """Support 'in' operator."""
        return key in self._data
    
    def get(self, key, default=None):
        """Support dict-like get method."""
        return self._data.get(key, default)
    
    def keys(self):
        """Support dict-like keys method."""
        return self._data.keys()
    
    def values_dict(self):
        """Support dict-like values method (renamed to avoid conflict with values property)."""
        return self._data.values()
    
    def items(self):
        """Support dict-like items method."""
        return self._data.items()
    
    def __repr__(self):
        """String representation of the QueryMatch."""
        chunk_preview = self.chunk[:50] if self.chunk else "None"
        return f"QueryMatch(chunk='{chunk_preview}...', score={self.score}, path='{self.path}')"


class QueryMatchTyped(TypedDict):
    """Single match from a semantic search query (TypedDict version)."""
    chunk: str  # Text chunk that matched the query
    path: str   # Document field path
    chunk_n: int  # Index of the chunk
    score: float  # Similarity score (0-1)
    document: dict  # Full document containing the match
    values: Optional[List[float]]  # Embedding vector values (optional)


class QueryResponse(TypedDict):
    """Complete response from a semantic search query (raw API response)."""
    matches: List[dict]  # Raw matches data from API


class QueryResponseTyped(TypedDict):
    """Complete response from a semantic search query (TypedDict version)."""
    matches: List[QueryMatchTyped]  # Matches sorted by relevance
