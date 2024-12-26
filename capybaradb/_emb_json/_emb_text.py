from typing import Optional, List, Dict, Any
from ._emb_models import EmbModels


class EmbText:
    SUPPORTED_EMB_MODELS = [
        EmbModels.TEXT_EMBEDDING_3_SMALL,
        EmbModels.TEXT_EMBEDDING_3_LARGE,
        EmbModels.TEXT_EMBEDDING_ADA_002,
    ]

    def __init__(
        self,
        text: str,
        chunks: List[str] = [],
        emb_model: str = "text-embedding-3-small",
        max_chunk_size: int = 200,
        chunk_overlap: int = 20,
        is_separator_regex: bool = False,
        separators: Optional[List[str]] = None,
        keep_separator: bool = False,
    ):
        if not self.is_valid_text(text):
            raise ValueError("Invalid text: must be a non-empty string.")

        if not self.is_valid_emb_model(emb_model):
            raise ValueError(f"Invalid embedding model: {emb_model} is not supported.")

        self.text = text
        self.chunks = chunks
        self.emb_model = emb_model
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.is_separator_regex = is_separator_regex
        self.separators = separators
        self.keep_separator = keep_separator

    def __repr__(self):
        return f'EmbText("{self.text}")'

    @staticmethod
    def is_valid_text(text: str) -> bool:
        return isinstance(text, str) and text.strip() != ""

    @classmethod
    def is_valid_emb_model(cls, emb_model: str) -> bool:
        return emb_model in cls.SUPPORTED_EMB_MODELS

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the EmbText instance to a JSON-serializable dictionary.
        """
        return {
            "@embText": {
                "text": self.text,
                "chunks": self.chunks,
                "emb_model": self.emb_model,
                "max_chunk_size": self.max_chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "is_separator_regex": self.is_separator_regex,
                "separators": self.separators,
                "keep_separator": self.keep_separator,
            }
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "EmbText":
        """
        Create an EmbText instance from a JSON-serializable dictionary.
        Defaults are applied if any properties are missing.
        """

        text = data.get("text")
        if text is None:
            raise ValueError("JSON data must include 'text' under '@embText'.")

        chunks = data.get("chunks", [])
        emb_model = data.get("emb_model", "text-embedding-3-small")
        max_chunk_size = data.get("max_chunk_size", 200)
        chunk_overlap = data.get("chunk_overlap", 20)
        is_separator_regex = data.get("is_separator_regex", False)
        separators = data.get("separators", None)
        keep_separator = data.get("keep_separator", False)

        return cls(
            text,
            chunks,
            emb_model,
            max_chunk_size,
            chunk_overlap,
            is_separator_regex,
            separators,
            keep_separator,
        )
