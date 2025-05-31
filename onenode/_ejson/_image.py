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
    ):
        """Initialize Image with base64-encoded image data."""
        if not self.is_valid_data(data):
            raise ValueError("Invalid data: must be a non-empty string containing valid base64-encoded image data.")
            
        # MIME type validation only matters when indexing
        # so we don't validate it here anymore

        self.data = data
        self.mime_type = mime_type
        self._chunks: List[str] = []  # Updated by the database
        self._url: Optional[str] = None  # URL is set by the server
        self.emb_model: Optional[str] = None
        self.vision_model: Optional[str] = None
        self.max_chunk_size: Optional[int] = None
        self.chunk_overlap: Optional[int] = None
        self.is_separator_regex: Optional[bool] = None
        self.separators: Optional[List[str]] = None
        self.keep_separator: Optional[bool] = None
        self.index_enabled: bool = False  # Default to False when index() isn't called

    def index(
        self,
        *,
        emb_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        is_separator_regex: Optional[bool] = None,
        separators: Optional[List[str]] = None,
        keep_separator: Optional[bool] = None,
    ) -> "Image":
        """Fluent builder method to enable indexing and set indexing parameters."""
        # Set index to True when this method is called
        self.index_enabled = True
        
        # MIME type validation happens here when indexing is enabled
        if not self.is_valid_mime_type(self.mime_type):
            supported_list = ", ".join(self.SUPPORTED_MIME_TYPES)
            raise ValueError(f"Unsupported mime type: '{self.mime_type}'. Supported types are: {supported_list}")
        
        if emb_model is not None:
            if not self.is_valid_emb_model(emb_model):
                supported_list = ", ".join(Models.TextToEmbedding.OpenAI.values())
                raise ValueError(f"Invalid embedding model: '{emb_model}' is not supported. Supported models are: {supported_list}")
            self.emb_model = emb_model
            
        if vision_model is not None:
            if not self.is_valid_vision_model(vision_model):
                supported_list = ", ".join(Models.ImageToText.OpenAI.values())
                raise ValueError(f"Invalid vision model: '{vision_model}' is not supported. Supported models are: {supported_list}")
            self.vision_model = vision_model
            
        if max_chunk_size is not None:
            self.max_chunk_size = max_chunk_size
            
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
            
        if is_separator_regex is not None:
            self.is_separator_regex = is_separator_regex
            
        if separators is not None:
            self.separators = separators
            
        if keep_separator is not None:
            self.keep_separator = keep_separator
            
        return self

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
                "index": self.index_enabled,  # Always include index flag
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
        
        # Create the instance with required parameters
        instance = cls(
            data=data_content,
            mime_type=mime_type,
        )
        
        # If index is true in the data, call index() to set up indexing
        if data.get("index", False):
            instance.index(
                emb_model=data.get("emb_model"),
                vision_model=data.get("vision_model"),
                max_chunk_size=data.get("max_chunk_size"),
                chunk_overlap=data.get("chunk_overlap"),
                is_separator_regex=data.get("is_separator_regex"),
                separators=data.get("separators"),
                keep_separator=data.get("keep_separator"),
            )
        # Otherwise just set the attributes without setting index_enabled=True
        else:
            if "emb_model" in data:
                instance.emb_model = data.get("emb_model")
            if "vision_model" in data:
                instance.vision_model = data.get("vision_model")
            if "max_chunk_size" in data:
                instance.max_chunk_size = data.get("max_chunk_size")
            if "chunk_overlap" in data:
                instance.chunk_overlap = data.get("chunk_overlap")
            if "is_separator_regex" in data:
                instance.is_separator_regex = data.get("is_separator_regex")
            if "separators" in data:
                instance.separators = data.get("separators")
            if "keep_separator" in data:
                instance.keep_separator = data.get("keep_separator")
        
        # Set chunks if they exist in the JSON
        if "chunks" in data:
            instance._chunks = data.get("chunks", [])
            
        # Set URL if it exists in the JSON
        if "url" in data:
            instance._url = data.get("url")
        
        return instance
