from typing import Optional, List, Dict, Any
from ._emb_models import EmbModels
from ._vision_models import VisionModels
import base64


class EmbImage:
    """
    EmbImage - A specialized data type for storing and processing images in CapybaraDB
    
    EmbImage enables multimodal capabilities by storing images that can be:
    1. Processed by vision models to extract textual descriptions
    2. Embedded for vector search (using the extracted descriptions)
    3. Stored alongside other document data
    
    When stored in the database, the image is processed asynchronously in the background:
    - If a vision model is specified, the image is analyzed to generate textual descriptions
    - If an embedding model is specified, these descriptions are embedded for semantic search
    - The results are stored in the 'chunks' property
    
    Usage:
        ```python
        from capybaradb import CapybaraDB, EmbImage, VisionModels
        import base64
        
        # Read an image file and convert to base64
        with open("path/to/image.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Create a document with an EmbImage field
        document = {
            "title": "Image Document",
            "image": EmbImage(
                image_data,
                vision_model=VisionModels.GPT_4O,  # For image understanding
                emb_model=None  # Can be None if only using vision model
            )
        }
        
        # Insert into CapybaraDB
        client = CapybaraDB()
        client.my_database.my_collection.insert([document])
        
        # Later, you can perform semantic searches that include image content
        ```
    """
    
    # List of supported embedding models for processing text chunks
    SUPPORTED_EMB_MODELS = [
        EmbModels.TEXT_EMBEDDING_3_SMALL,
        EmbModels.TEXT_EMBEDDING_3_LARGE,
        EmbModels.TEXT_EMBEDDING_ADA_002,
    ]
    
    # List of supported vision models for analyzing images
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
        """
        Initialize an EmbImage object for image storage and processing.
        
        Args:
            data: Base64-encoded image data. Must be a non-empty string.
            
            emb_model: The embedding model to use for text chunks. 
                      Can be None if only using vision model.
                      
            vision_model: The vision model to use for analyzing the image.
                         Can be None if only storing the image.
                         
            max_chunk_size: Maximum character length for each text chunk.
                           Used when processing vision model output.
                           
            chunk_overlap: Number of overlapping characters between consecutive chunks.
                          
            is_separator_regex: Whether to treat separators as regex patterns.
            
            separators: List of separator strings or regex patterns.
            
            keep_separator: If True, separators remain in the chunked text.
        
        Raises:
            ValueError: If the data is not a valid string or if the models are not supported.
        """
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
    def is_valid_emb_model(cls, emb_model: Optional[str]) -> bool:
        return emb_model is None or emb_model in cls.SUPPORTED_EMB_MODELS

    @classmethod
    def is_valid_vision_model(cls, vision_model: Optional[str]) -> bool:
        return vision_model is None or vision_model in cls.SUPPORTED_VISION_MODELS

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
