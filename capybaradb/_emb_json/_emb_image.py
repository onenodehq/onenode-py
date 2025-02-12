from typing import Optional, List, Dict, Any
from ._emb_models import EmbModels
from ._vision_models import VisionModels
import base64


class EmbImage:
    SUPPORTED_EMB_MODELS = [
        EmbModels.TEXT_EMBEDDING_3_SMALL,
        EmbModels.TEXT_EMBEDDING_3_LARGE,
        EmbModels.TEXT_EMBEDDING_ADA_002,
    ]
    SUPPORTED_VISION_MODELS = [
        VisionModels.GPT_4O_MINI,
        VisionModels.GPT_4O,
        VisionModels.GPT_4O_TURBO,
        VisionModels.GPT_O1,
    ]

    def __init__(
        self,
        data: str,  # base64 encoded image (change this if needed)
        emb_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        is_separator_regex: Optional[bool] = None,
        separators: Optional[List[str]] = None,
        keep_separator: Optional[bool] = None,
    ):
        if not self.is_valid_data(data):
            raise ValueError("Invalid data: must be a non-empty string.")

        if not self.is_valid_emb_model(emb_model):
            raise ValueError(f"Invalid embedding model: {emb_model} is not supported.")

        if not self.is_valid_vision_model(vision_model):
            raise ValueError(f"Invalid vision model: {vision_model} is not supported.")

        self.data = data
        self._chunks: List[str] = []  # Private attribute: updated only internally.
        self.emb_model = emb_model
        self.vision_model = vision_model
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.is_separator_regex = is_separator_regex
        self.separators = separators
        self.keep_separator = keep_separator

    def __repr__(self):
        if self._chunks:
            return f'EmbImage("{self._chunks[0]}")'
        # Alternative representation when chunks are not set
        return "EmbImage(<raw data>)"

    @property
    def chunks(self) -> List[str]:
        """Read-only property for chunks."""
        return self._chunks

    @staticmethod
    def is_valid_data(data: str) -> bool:
        if not (isinstance(data, str) and data.strip() != ""):
            return False
        try:
            # Validate that data is a valid base64 encoded string.
            base64.b64decode(data, validate=True)
            return True
        except Exception:
            return False

    @classmethod
    def is_valid_emb_model(cls, emb_model: str) -> bool:
        return emb_model in cls.SUPPORTED_EMB_MODELS

    @classmethod
    def is_valid_vision_model(cls, vision_model: str) -> bool:
        return vision_model in cls.SUPPORTED_VISION_MODELS

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the EmbImage instance to a JSON-serializable dictionary.
        """
        return {
            "@embImage": {
                "data": self.data,
                "chunks": self._chunks,
                "emb_model": self.emb_model,
                "vision_model": self.vision_model,
                "max_chunk_size": self.max_chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "is_separator_regex": self.is_separator_regex,
                "separators": self.separators,
                "keep_separator": self.keep_separator,
            }
        }

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "EmbImage":
        """
        Create an EmbImage instance from a JSON-serializable dictionary.
        Defaults are applied if any properties are missing.
        Assumes the input dictionary is the inner dictionary (i.e., the value under "@embImage").
        """
        image_data = json_dict.get("data")
        if image_data is None:
            raise ValueError("JSON data must include 'data' under '@embImage'.")

        emb_model = json_dict.get("emb_model")
        vision_model = json_dict.get("vision_model")
        max_chunk_size = json_dict.get("max_chunk_size")
        chunk_overlap = json_dict.get("chunk_overlap")
        is_separator_regex = json_dict.get("is_separator_regex")
        separators = json_dict.get("separators")
        keep_separator = json_dict.get("keep_separator")

        instance = cls(
            image_data,
            emb_model,
            vision_model,
            max_chunk_size,
            chunk_overlap,
            is_separator_regex,
            separators,
            keep_separator,
        )
        instance._chunks = json_dict.get("chunks", [])
        return instance
