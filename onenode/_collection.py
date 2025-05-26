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
import json
from ._types import QueryResponse
from ._ejson._text import Text
from ._ejson._image import Image

# Serialization for BSON types
BSON_SERIALIZERS = {
    Text: lambda v: {"xText": v.to_json()},
    Image: lambda v: {"xImage": v.to_json()},
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
    """Base class for all API client-related errors."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class AuthenticationError(APIClientError):
    """Error raised for authentication-related issues."""
    pass


class ClientRequestError(APIClientError):
    """Error raised for client-side issues such as validation errors."""
    pass


class ServerError(APIClientError):
    """Error raised for server-side issues."""
    pass


class Collection:
    """Collection in OneNode for document operations and semantic search."""
    
    def __init__(
        self, api_key: str, project_id: str, db_name: str, collection_name: str
    ):
        """Initialize collection instance."""
        self.api_key = api_key
        self.project_id = project_id
        self.db_name = db_name
        self.collection_name = collection_name

    def get_collection_url(self) -> str:
        return f"https://api.onenode.ai/v1/db/{self.project_id}_{self.db_name}/collection/{self.collection_name}/document"

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def __serialize(self, value):
        """Serialize BSON types, Text, and nested structures into JSON-compatible formats."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, dict):
            return {k: self.__serialize(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self.__serialize(item) for item in value]

        if isinstance(value, Text):
            return value.to_json()
            
        if isinstance(value, Image):
            return value.to_json()

        serializer = BSON_SERIALIZERS.get(type(value))
        if serializer:
            return serializer(value)

        raise TypeError(f"Unsupported BSON type: {type(value)}")

    def __deserialize(self, value, depth=0):
        """Convert JSON-compatible structures back to BSON types and Text."""
        if isinstance(value, dict):
            for key in value:
                if "xText" in value:
                    return Text.from_json(value)
                if "xImage" in value:
                    return Image.from_json(value)
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
        """Insert documents into the collection."""
        url = self.get_collection_url()
        headers = self.get_headers()
        serialized_docs = [self.__serialize(doc) for doc in documents]
        
        files = {}
        data = {"documents": json.dumps(serialized_docs)}

        response = requests.post(url, headers=headers, files=files, data=data)
        return self.handle_response(response)

    def update(self, filter: dict, update: dict, upsert: bool = False) -> dict:
        """Update documents matching filter."""
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_filter = self.__serialize(filter)
        transformed_update = self.__serialize(update)
        
        files = {}
        data = {
            "filter": json.dumps(transformed_filter),
            "update": json.dumps(transformed_update),
            "upsert": str(upsert).lower(),
        }

        response = requests.put(url, headers=headers, files=files, data=data)
        return self.handle_response(response)

    def delete(self, filter: dict) -> dict:
        """Delete documents matching filter."""
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_filter = self.__serialize(filter)
        
        files = {}
        data = {"filter": json.dumps(transformed_filter)}

        response = requests.delete(url, headers=headers, files=files, data=data)
        return self.handle_response(response)

    def find(
        self,
        filter: dict,
        projection: dict = None,
        sort: dict = None,
        limit: int = None,
        skip: int = None,
    ) -> list[dict]:
        """Find documents matching filter."""
        url = f"{self.get_collection_url()}/find"
        headers = self.get_headers()
        transformed_filter = self.__serialize(filter)
        
        files = {}
        data = {"filter": json.dumps(transformed_filter)}
        
        if projection is not None:
            data["projection"] = json.dumps(projection)
        if sort is not None:
            data["sort"] = json.dumps(sort)
        if limit is not None:
            data["limit"] = str(limit)
        if skip is not None:
            data["skip"] = str(skip)

        response = requests.post(url, headers=headers, files=files, data=data)
        response_data = self.handle_response(response)
        return response_data.get("docs", [])

    def query(
        self,
        query: str,
        filter: dict = None,
        projection: dict = None,
        emb_model: str = None,
        top_k: int = None,
        include_values: bool = None,
    ) -> QueryResponse:
        """Perform semantic search on the collection."""
        url = f"{self.get_collection_url()}/query"
        headers = self.get_headers()

        files = {}
        data = {"query": query}
        
        if filter is not None:
            data["filter"] = json.dumps(self.__serialize(filter))
        if projection is not None:
            data["projection"] = json.dumps(projection)
        if emb_model is not None:
            data["emb_model"] = emb_model
        if top_k is not None:
            data["top_k"] = str(top_k)
        if include_values is not None:
            data["include_values"] = str(include_values).lower()

        response = requests.post(url, headers=headers, files=files, data=data)
        response_data = self.handle_response(response)
        return response_data.get("matches", [])

    def drop(self) -> None:
        """Delete the entire collection."""
        url = f"https://api.onenode.ai/v1/db/{self.project_id}_{self.db_name}/collection/{self.collection_name}"
        headers = self.get_headers()
        
        files = {}
        data = {}
        
        response = requests.delete(url, headers=headers, files=files, data=data)
        if response.status_code == 204:
            return None
            
        self.handle_response(response)
