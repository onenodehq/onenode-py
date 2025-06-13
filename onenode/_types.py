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
    - match.embedding - Embedding vector embedding (optional, when include_embedding=True)
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
    def embedding(self) -> Optional[List[float]]:
        """Embedding vector embedding (only present when include_embedding=True)."""
        return self._data.get('embedding')
    
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
    embedding: Optional[List[float]]  # Embedding vector embedding (optional)


class InsertResponse:
    """Insert operation response with attribute-style access.
    
    Provides dot notation access for fixed fields:
    - response.inserted_ids - List of inserted document IDs
    """
    
    def __init__(self, data: dict):
        """Initialize InsertResponse with raw response data."""
        self._data = data
    
    @property
    def inserted_ids(self) -> List[str]:
        """List of inserted document IDs."""
        return self._data.get('inserted_ids', [])
    
    def __repr__(self):
        """String representation of the InsertResponse."""
        count = len(self.inserted_ids)
        return f"InsertResponse(inserted_ids={count} documents)"
