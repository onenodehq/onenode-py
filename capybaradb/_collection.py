from bson import (
    Code,
    MaxKey,
    MinKey,
    Regex,
    Timestamp,
    ObjectId,
    Decimal128,
    Binary,
)
from datetime import datetime
import requests
from ._types import QueryResponse
from ._emb_json._emb_text import EmbText
from ._emb_json._emb_image import EmbImage

# Map specific BSON types to their serialization logic
# This mapping enables automatic conversion of complex data types when sending data to CapybaraDB:
# - EmbText: Special CapybaraDB type for text that will be automatically embedded
# - ObjectId: MongoDB-style unique identifiers
# - datetime: Python datetime objects
# - Decimal128: High-precision decimal numbers
# - Binary: Binary data
# - Regex: Regular expressions
# - Code: JavaScript code
# - Timestamp: Precise timestamps
# - MinKey/MaxKey: Special BSON types for comparison operations
BSON_SERIALIZERS = {
    EmbText: lambda v: {"@embText": v.to_json()},
    ObjectId: lambda v: {"$oid": str(v)},
    datetime: lambda v: {"$date": v.isoformat()},
    Decimal128: lambda v: {"$numberDecimal": str(v)},
    Binary: lambda v: {"$binary": v.hex()},
    Regex: lambda v: {"$regex": v.pattern, "$options": v.flags},
    Code: lambda v: {"$code": str(v)},
    Timestamp: lambda v: {"$timestamp": {"t": v.time, "i": v.inc}},
    MinKey: lambda v: {"$minKey": 1},
    MaxKey: lambda v: {"$maxKey": 1},
}


class APIClientError(Exception):
    """
    Base class for all API client-related errors.
    
    Provides a foundation for more specific error types with status codes and messages.
    All API errors extend from this class.
    """

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class AuthenticationError(APIClientError):
    """
    Error raised for authentication-related issues.
    
    Thrown when there are problems with API keys or authentication tokens.
    Typically occurs with status code 401.
    """

    pass


class ClientRequestError(APIClientError):
    """
    Error raised for client-side issues such as validation errors.
    
    Thrown for problems like invalid parameters, missing required fields,
    or other client-side validation errors.
    """

    pass


class ServerError(APIClientError):
    """
    Error raised for server-side issues.
    
    Thrown when the CapybaraDB service encounters internal errors,
    is unavailable, or otherwise cannot process a valid request.
    """

    pass


class Collection:
    """
    Collection - Represents a collection in CapybaraDB
    
    The Collection class is the primary interface for performing operations on documents:
    - insert: Add new documents to the collection
    - update: Modify existing documents
    - delete: Remove documents
    - find: Retrieve documents based on filters
    - query: Perform semantic searches on embedded text fields
    
    Collections in CapybaraDB are similar to collections in MongoDB or tables in SQL databases.
    They store documents (JSON objects) that can contain embedded text fields for semantic search.
    
    This class handles:
    1. Serialization of complex data types (BSON, EmbText) for API transmission
    2. Deserialization of API responses back into appropriate Python types
    3. Error handling with specific error types
    4. HTTP communication with the CapybaraDB API
    
    Usage:
        ```python
        from capybaradb import CapybaraDB, EmbText
        
        client = CapybaraDB()
        collection = client.db("my_database").collection("my_collection")
        
        # Insert documents
        collection.insert([
            { 
                "title": "Document with embedded text",
                "content": EmbText("This text will be automatically embedded for semantic search")
            }
        ])
        
        # Find documents by exact match
        docs = collection.find({"title": "Document with embedded text"})
        
        # Perform semantic search
        results = collection.query("embedded text search")
        ```
        
        # Using EmbText for semantic text search:
        ```python
        from capybaradb import CapybaraDB, EmbText, EmbModels
        
        # Create a document with EmbText for automatic text embedding
        doc = {
            "title": "Article about AI",
            "content": EmbText(
                "Artificial intelligence is transforming industries worldwide.",
                emb_model=EmbModels.TEXT_EMBEDDING_3_SMALL,  # Optional: specify embedding model
                max_chunk_size=200,                          # Optional: control chunking
                chunk_overlap=20                             # Optional: overlap between chunks
            )
        }
        collection.insert([doc])
        
        # Later, query for semantically similar content
        results = collection.query("AI technology impact")
        ```
        
        # Using EmbImage for multimodal capabilities:
        ```python
        from capybaradb import CapybaraDB, EmbImage, VisionModels
        import base64
        
        # Read and encode an image
        with open("path/to/image.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Create a document with EmbImage for vision model processing
        doc = {
            "title": "Product Image",
            "description": "Our latest product",
            "image": EmbImage(
                image_data,                                  # Base64-encoded image
                vision_model=VisionModels.GPT_4O,            # Vision model for analysis
                emb_model=EmbModels.TEXT_EMBEDDING_3_SMALL   # Optional: for embedding descriptions
            )
        }
        collection.insert([doc])
        
        # Later, query for images with similar content
        results = collection.query("product with blue background")
        ```
    """
    
    def __init__(
        self, api_key: str, project_id: str, db_name: str, collection_name: str
    ):
        """
        Creates a new Collection instance.
        
        Note: You typically don't need to create this directly.
        Instead, use the `collection()` method on a Database instance.
        
        Args:
            api_key: API key for authentication
            project_id: Project ID that identifies your CapybaraDB project
            db_name: Name of the database containing this collection
            collection_name: Name of this collection
        """
        self.api_key = api_key
        self.project_id = project_id
        self.db_name = db_name
        self.collection_name = collection_name

    def get_collection_url(self) -> str:
        return f"https://api.capybaradb.co/v0/db/{self.project_id}_{self.db_name}/collection/{self.collection_name}/document"

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def __serialize(self, value):
        """
        Efficiently serialize BSON types, EmbText, and nested structures into JSON-compatible formats.
        """
        # Early return for primitive JSON-compatible types
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        # Handle dictionaries (fast path)
        if isinstance(value, dict):
            return {k: self.__serialize(v) for k, v in value.items()}

        # Handle lists (fast path)
        if isinstance(value, list):
            return [self.__serialize(item) for item in value]

        if isinstance(value, EmbText):
            return value.to_json()
            
        if isinstance(value, EmbImage):
            return value.to_json()

        # Check if the value matches a BSON-specific type
        serializer = BSON_SERIALIZERS.get(type(value))
        if serializer:
            return serializer(value)

        # Fallback for unsupported types
        raise TypeError(f"Unsupported BSON type: {type(value)}")

    def __deserialize(self, value, depth=0):
        """
        Recursively convert JSON-compatible structures back to BSON types and EmbText.
        """
        if isinstance(value, dict):
            # Quickly check if any keys indicate BSON special types
            for key in value:
                if "@embText" in value:
                    return EmbText.from_json(value["@embText"])
                if "@embImage" in value:
                    return EmbImage.from_json(value["@embImage"])
                elif key.startswith("$"):
                    if key == "$oid":
                        return ObjectId(value["$oid"])
                    if key == "$date":
                        return datetime.fromisoformat(value["$date"])
                    if key == "$numberDecimal":
                        return Decimal128(value["$numberDecimal"])
                    if key == "$binary":
                        return Binary(bytes.fromhex(value["$binary"]))
                    if key == "$regex":
                        return Regex(value["$regex"], value.get("$options", 0))
                    if key == "$code":
                        return Code(value["$code"])
                    if key == "$timestamp":
                        return Timestamp(
                            value["$timestamp"]["t"], value["$timestamp"]["i"]
                        )
                    if key == "$minKey":
                        return MinKey()
                    if key == "$maxKey":
                        return MaxKey()

            # Fallback: Regular recursive deserialization for non-BSON keys
            return {k: self.__deserialize(v, depth + 1) for k, v in value.items()}

        elif isinstance(value, list):
            return [self.__deserialize(item, depth + 1) for item in value]

        elif value is None:
            return None

        elif isinstance(value, (bool, int, float, str)):
            return value

        else:
            raise TypeError(
                f"Unsupported BSON type during deserialization: {type(value)}"
            )

    def handle_response(self, response):
        try:
            response.raise_for_status()
            json_response = response.json()
            return self.__deserialize(json_response)
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                status = error_data.get("status", "error")
                code = error_data.get("code", 500)
                message = error_data.get("message", "An unknown error occurred.")

                if code == 401:
                    raise AuthenticationError(code, message) from e
                elif code >= 400 and code < 500:
                    raise ClientRequestError(code, message) from e
                else:
                    raise ServerError(code, message) from e

            except ValueError:
                raise APIClientError(response.status_code, response.text) from e

    def insert(self, documents: list[dict]) -> dict:
        """
        Insert one or more documents into the collection.
        
        Note: When inserting documents with EmbText or EmbImage fields, there will be a short delay
        (typically a few seconds) before these documents become available for semantic search.
        This is because CapybaraDB processes embeddings asynchronously on the server side after
        the document is stored.
        
        Args:
            documents: List of documents to insert
            
        Returns:
            Dictionary with insertion result information
        """
        url = self.get_collection_url()
        headers = self.get_headers()
        serialized_docs = [self.__serialize(doc) for doc in documents]
        data = {"documents": serialized_docs}

        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)

    def update(self, filter: dict, update: dict, upsert: bool = False) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_filter = self.__serialize(filter)
        transformed_update = self.__serialize(update)
        data = {
            "filter": transformed_filter,
            "update": transformed_update,
            "upsert": upsert,
        }

        response = requests.put(url, headers=headers, json=data)
        return self.handle_response(response)

    def delete(self, filter: dict) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_filter = self.__serialize(filter)
        data = {"filter": transformed_filter}

        response = requests.delete(url, headers=headers, json=data)
        return self.handle_response(response)

    def find(
        self,
        filter: dict,
        projection: dict = None,
        sort: dict = None,
        limit: int = None,
        skip: int = None,
    ) -> list[dict]:
        url = f"{self.get_collection_url()}/find"
        headers = self.get_headers()
        transformed_filter = self.__serialize(filter)
        data = {
            "filter": transformed_filter,
            "projection": projection,
            "sort": sort,
            "limit": limit,
            "skip": skip,
        }

        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)

    def query(
        self,
        query: str,
        filter: dict = None,
        projection: dict = None,
        emb_model: str = None,
        top_k: int = None,
        include_values: bool = None,
    ) -> QueryResponse:
        """
        Perform a semantic search query on the collection.
        
        This method searches for documents that semantically match the given query text.
        It returns documents containing EmbText or EmbImage fields that are semantically
        similar to the query, sorted by relevance score.
        
        Args:
            query: The text to search for
            filter: Optional filter to apply to documents before semantic search (MongoDB-style query filter)
            projection: Optional specification of which fields to include or exclude in the response
            emb_model: Optional embedding model to use for the query (default: "text-embedding-3-small")
            top_k: Optional maximum number of results to return (default: 10)
            include_values: Optional flag to include vector values in the response (default: False)
            
        Returns:
            QueryResponse object containing matches sorted by relevance
            
        Example:
            ```python
            # Basic query
            results = collection.query("machine learning techniques")
            
            # Advanced query with filter
            results = collection.query(
                "machine learning techniques",
                filter={"category": "AI", "published": True},
                top_k=5
            )
            ```
        """
        url = f"{self.get_collection_url()}/query"
        headers = self.get_headers()

        data = {"query": query}
        if filter is not None:
            data["filter"] = filter
        if projection is not None:
            data["projection"] = projection
        if emb_model is not None:
            data["emb_model"] = emb_model
        if top_k is not None:
            data["top_k"] = top_k
        if include_values is not None:
            data["include_values"] = include_values

        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)
