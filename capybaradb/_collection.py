from bson import ObjectId
import requests
from ._emb_json._emb_text import EmbText


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

    def __jsonify_embjson(self, value):
        """
        Recursively convert all EmbText and ObjectId instances to JSON,
        handling nested dictionaries and lists comprehensively.
        """
        if isinstance(value, (EmbText, ObjectId)):
            return value.to_json()
        elif isinstance(value, dict):
            return {k: self.__jsonify_embjson(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.__jsonify_embjson(item) for item in value]
        else:
            return value

    def handle_response(self, response):
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                # Attempt to parse the JSON error response
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
                # Response is not JSON; use the raw text
                raise APIClientError(response.status_code, response.text) from e

    def insert(self, documents: list[dict]) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_documents = [self.__jsonify_embjson(doc) for doc in documents]
        data = {"documents": transformed_documents}

        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)

    def update(self, filter: dict, update: dict, upsert: bool = False) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_filter = self.__jsonify_embjson(filter)
        transformed_update = self.__jsonify_embjson(update)
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
        transformed_filter = self.__jsonify_embjson(filter)
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
        transformed_filter = self.__jsonify_embjson(filter)
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
    ) -> list[dict]:
        url = f"{self.get_collection_url()}/query"
        headers = self.get_headers()

        # Create the data dictionary with only non-None values
        data = {"query": query}  # 'query' is required
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
