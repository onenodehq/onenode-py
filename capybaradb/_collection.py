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

# Map specific BSON types to their serialization logic
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
    def __init__(
        self, api_key: str, project_id: str, db_name: str, collection_name: str
    ):
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
        emb_model: str = None,
        top_k: int = None,
        include_values: bool = None,
        projection: dict = None,
    ) -> QueryResponse:
        url = f"{self.get_collection_url()}/query"
        headers = self.get_headers()

        data = {"query": query}
        if emb_model is not None:
            data["emb_model"] = emb_model
        if top_k is not None:
            data["top_k"] = top_k
        if include_values is not None:
            data["include_values"] = include_values
        if projection is not None:
            data["projection"] = projection

        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)
