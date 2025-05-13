from typing import Optional, List, Dict, Any
from ._models import Models
import base64


class Image:
    """Specialized data type for images with vision model processing."""
    
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
        emb_model: Optional[str] = Models.TextToEmbedding.OpenAI.TEXT_EMBEDDING_3_SMALL,
        vision_model: Optional[str] = Models.ImageToText.OpenAI.GPT_4O_MINI,
        max_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        is_separator_regex: Optional[bool] = None,
        separators: Optional[List[str]] = None,
        keep_separator: Optional[bool] = None,
    ):
        """Initialize Image with base64-encoded image data."""
        if not self.is_valid_data(data):
            raise ValueError("Invalid data: must be a non-empty string containing valid base64-encoded image data.")
            
        if not self.is_valid_mime_type(mime_type):
            supported_list = ", ".join(self.SUPPORTED_MIME_TYPES)
            raise ValueError(f"Unsupported mime type: '{mime_type}'. Supported types are: {supported_list}")

        if emb_model is not None and not self.is_valid_emb_model(emb_model):
            supported_list = ", ".join(Models.TextToEmbedding.OpenAI.values())
            raise ValueError(f"Invalid embedding model: '{emb_model}' is not supported. Supported models are: {supported_list}")

        if vision_model is not None and not self.is_valid_vision_model(vision_model):
            supported_list = ", ".join(Models.ImageToText.OpenAI.values())
            raise ValueError(f"Invalid vision model: '{vision_model}' is not supported. Supported models are: {supported_list}")

        self.data = data
        self.mime_type = mime_type
        self._chunks: List[str] = []  # Updated by the database
        self._url: Optional[str] = None  # URL is set by the server
        self.emb_model = emb_model
        self.vision_model = vision_model
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.is_separator_regex = is_separator_regex
        self.separators = separators
        self.keep_separator = keep_separator

    def __repr__(self):
        if self._url:
            return f'Image({self._url})'
        if self._chunks:
            return f'Image("{self._chunks[0]}")'
        return "Image(<raw data>)"

    @property
    def chunks(self) -> List[str]:
        """Read-only property for chunks."""
        return self._chunks
        
    @property
    def url(self) -> Optional[str]:
        """Read-only property for the URL of the image (set by server)."""
        return self._url

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
        return emb_model is None or emb_model in Models.TextToEmbedding.OpenAI.values()

    @classmethod
    def is_valid_vision_model(cls, vision_model: Optional[str]) -> bool:
        """Check if vision model is supported."""
        return vision_model is None or vision_model in Models.ImageToText.OpenAI.values()

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        # Start with required fields
        result = {
            "xImage": {
                "data": self.data,
                "mime_type": self.mime_type,
            }
        }
        
        # Only include chunks if they exist
        if self._chunks:
            result["xImage"]["chunks"] = self._chunks
            
        # Include URL if it exists
        if self._url:
            result["xImage"]["url"] = self._url

        # Add other fields only if they are not None
        if self.emb_model is not None:
            result["xImage"]["emb_model"] = self.emb_model
        if self.vision_model is not None:
            result["xImage"]["vision_model"] = self.vision_model
        if self.max_chunk_size is not None:
            result["xImage"]["max_chunk_size"] = self.max_chunk_size
        if self.chunk_overlap is not None:
            result["xImage"]["chunk_overlap"] = self.chunk_overlap
        if self.is_separator_regex is not None:
            result["xImage"]["is_separator_regex"] = self.is_separator_regex
        if self.separators is not None:
            result["xImage"]["separators"] = self.separators
        if self.keep_separator is not None:
            result["xImage"]["keep_separator"] = self.keep_separator
            
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Image":
        """Create Image from JSON dictionary."""
        # Check if the data is wrapped with 'xImage'
        if "xImage" in data:
            data = data["xImage"]
            
        if "mime_type" not in data:
            raise ValueError("JSON data must include 'mime_type' under 'xImage'.")
        
        # Get optional fields with their defaults
        data_content = data.get("data")
        mime_type = data.get("mime_type")
        emb_model = data.get("emb_model")
        vision_model = data.get("vision_model")
        max_chunk_size = data.get("max_chunk_size")
        chunk_overlap = data.get("chunk_overlap")
        is_separator_regex = data.get("is_separator_regex")
        separators = data.get("separators")
        keep_separator = data.get("keep_separator")
        
        # Create the instance
        instance = cls(
            data=data_content,
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
        if "chunks" in data:
            instance._chunks = data.get("chunks", [])
            
        # Set URL if it exists in the JSON
        if "url" in data:
            instance._url = data.get("url")
        
        return instance
