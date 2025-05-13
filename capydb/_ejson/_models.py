class Models:
    """All supported model constants, grouped by task and vendor."""

    class TextToEmbedding:
        """Embedding models, by vendor."""
        class OpenAI:
            TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
            TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
            TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    class ImageToText:
        """Vision‑to‑text models, by vendor."""
        class OpenAI:
            GPT_4O = "gpt-4o"
            GPT_4O_MINI = "gpt-4o-mini"
            GPT_4_TURBO = "gpt-4-turbo"
            O1 = "o1"