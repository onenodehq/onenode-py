class EmbModels:
    """
    EmbModels - Constants for supported embedding models in CapybaraDB
    
    This class provides constants for the embedding models supported by CapybaraDB.
    These models are used to generate vector embeddings from text for semantic search.
    
    When creating an EmbText object, you can specify which embedding model to use.
    Different models have different characteristics in terms of:
    - Embedding quality (semantic understanding)
    - Vector dimensions
    - Processing speed
    - Cost
    
    Usage:
        ```python
        from capybaradb import EmbText, EmbModels
        
        # Create an EmbText with a specific embedding model
        text = EmbText(
            "This text will be embedded using the specified model",
            emb_model=EmbModels.TEXT_EMBEDDING_3_SMALL
        )
        ```
    """
    
    # OpenAI's text-embedding-3-small model
    # A smaller, faster embedding model with 1536 dimensions.
    # Good balance of quality and performance for most use cases.
    # This is the default model used if none is specified.
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    
    # OpenAI's text-embedding-3-large model
    # A larger, more powerful embedding model with 3072 dimensions.
    # Provides higher quality embeddings for more nuanced semantic search,
    # but is more computationally intensive.
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    
    # OpenAI's text-embedding-ada-002 model (legacy)
    # The previous generation embedding model with 1536 dimensions.
    # Included for backward compatibility with existing applications.
    # For new applications, prefer the text-embedding-3 models.
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
