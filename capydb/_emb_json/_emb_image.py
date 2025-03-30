from typing import Optional, List, Dict, Any
from ._emb_models import EmbModels
from ._vision_models import VisionModels
import base64


class EmbImage:
    """Specialized data type for images with vision model processing."""
    
    # Supported embedding models
    SUPPORTED_EMB_MODELS = [
        EmbModels.TEXT_EMBEDDING_3_SMALL,
        EmbModels.TEXT_EMBEDDING_3_LARGE,
        EmbModels.TEXT_EMBEDDING_ADA_002,
    ]
    
    # Supported vision models
    SUPPORTED_VISION_MODELS = [
        VisionModels.GPT_4O_MINI,
        VisionModels.GPT_4O,
        VisionModels.GPT_4O_TURBO,
        VisionModels.GPT_O1,
    ]
    
    # Supported mime types
    SUPPORTED_MIME_TYPES = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/webp",
    ]

    def __init__(
        self,
        data: str,  # base64 encoded image
        mime_type: str,  # mime type of the image
        emb_model: Optional[str] = EmbModels.TEXT_EMBEDDING_3_SMALL,
        vision_model: Optional[str] = VisionModels.GPT_4O_MINI,
        max_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        is_separator_regex: Optional[bool] = None,
        separators: Optional[List[str]] = None,
        keep_separator: Optional[bool] = None,
    ):
        """Initialize EmbImage with base64-encoded image data."""
        if not self.is_valid_data(data):
            raise ValueError("Invalid data: must be a non-empty string containing valid base64-encoded image data.")
            
        if not self.is_valid_mime_type(mime_type):
            supported_list = ", ".join(self.SUPPORTED_MIME_TYPES)
            raise ValueError(f"Unsupported mime type: '{mime_type}'. Supported types are: {supported_list}")

        if emb_model is not None and not self.is_valid_emb_model(emb_model):
            supported_list = ", ".join(self.SUPPORTED_EMB_MODELS)
            raise ValueError(f"Invalid embedding model: '{emb_model}' is not supported. Supported models are: {supported_list}")

        if vision_model is not None and not self.is_valid_vision_model(vision_model):
            supported_list = ", ".join(self.SUPPORTED_VISION_MODELS)
            raise ValueError(f"Invalid vision model: '{vision_model}' is not supported. Supported models are: {supported_list}")

        self.data = data
        self.mime_type = mime_type
        self._chunks: List[str] = []  # Updated by the database
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
        return "EmbImage(<raw data>)"

    @property
    def chunks(self) -> List[str]:
        """Read-only property for chunks."""
        return self._chunks

    @staticmethod
    def is_valid_data(data: str) -> bool:
        """Validate data is valid base64-encoded string."""
        if not (isinstance(data, str) and data.strip() != ""):
            return False
        try:
            base64.b64decode(data, validate=True)
            return True
        except Exception:
            return False
            
    @classmethod
    def is_valid_mime_type(cls, mime_type: str) -> bool:
        """Check if mime_type is supported."""
        return mime_type in cls.SUPPORTED_MIME_TYPES

    @classmethod
    def is_valid_emb_model(cls, emb_model: Optional[str]) -> bool:
        """Check if embedding model is supported."""
        return emb_model is None or emb_model in cls.SUPPORTED_EMB_MODELS

    @classmethod
    def is_valid_vision_model(cls, vision_model: Optional[str]) -> bool:
        """Check if vision model is supported."""
        return vision_model is None or vision_model in cls.SUPPORTED_VISION_MODELS

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        # Start with required fields
        result = {
            "data": self.data,
            "mime_type": self.mime_type,
        }
        
        # Only include chunks if they exist
        if self._chunks:
            result["chunks"] = self._chunks
        
        # Add other fields only if they are not None
        if self.emb_model is not None:
            result["emb_model"] = self.emb_model
        if self.vision_model is not None:
            result["vision_model"] = self.vision_model
        if self.max_chunk_size is not None:
            result["max_chunk_size"] = self.max_chunk_size
        if self.chunk_overlap is not None:
            result["chunk_overlap"] = self.chunk_overlap
        if self.is_separator_regex is not None:
            result["is_separator_regex"] = self.is_separator_regex
        if self.separators is not None:
            result["separators"] = self.separators
        if self.keep_separator is not None:
            result["keep_separator"] = self.keep_separator
            
        return result

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "EmbImage":
        """Create EmbImage from JSON dictionary."""
        # Check for required fields
        if "data" not in json_dict:
            raise ValueError("Missing required field 'data'")
        if "mime_type" not in json_dict:
            raise ValueError("Missing required field 'mime_type'")
        
        # Get optional fields with their defaults
        data = json_dict.get("data")
        mime_type = json_dict.get("mime_type")
        emb_model = json_dict.get("emb_model")
        vision_model = json_dict.get("vision_model")
        max_chunk_size = json_dict.get("max_chunk_size")
        chunk_overlap = json_dict.get("chunk_overlap")
        is_separator_regex = json_dict.get("is_separator_regex")
        separators = json_dict.get("separators")
        keep_separator = json_dict.get("keep_separator")
        
        # Create the instance
        instance = cls(
            data=data,
            mime_type=mime_type,
            emb_model=emb_model,
            vision_model=vision_model,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=is_separator_regex,
            separators=separators,
            keep_separator=keep_separator,
        )
        
        # Set chunks if they exist in the JSON
        if "chunks" in json_dict:
            instance._chunks = json_dict["chunks"]
        
        return instance
