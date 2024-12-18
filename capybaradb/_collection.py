import requests
from ._emb_json._emb_text import EmbText


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

    def transform_emb_text(self, document: dict) -> dict:
        """
        Recursively traverse the document and convert EmbText instances to JSON.
        """
        for key, value in document.items():
            if isinstance(value, EmbText):
                document[key] = value.to_json()  # Convert EmbText to JSON
            elif isinstance(value, dict):
                document[key] = self.transform_emb_text(
                    value
                )  # Recurse for nested dicts
            elif isinstance(value, list):
                document[key] = [
                    self.transform_emb_text(item) if isinstance(item, dict) else item
                    for item in value
                ]  # Handle lists of dicts or other values
        return document

    def handle_response(self, response):
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_message = response.json().get("error", response.text)
            except ValueError:
                error_message = response.text  # If response is not JSON
            raise RuntimeError(
                f"HTTP Error: {e}. Server message: {error_message}"
            ) from e

    def insert(self, documents: list[dict]) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_documents = [self.transform_emb_text(doc) for doc in documents]
        data = {"documents": transformed_documents}

        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)

    def update(self, filter: dict, update: dict, upsert: bool = False) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_update = self.transform_emb_text(update)
        data = {"filter": filter, "update": transformed_update, "upsert": upsert}

        response = requests.put(url, headers=headers, json=data)
        return self.handle_response(response)

    def delete(self, filter: dict) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        data = {"filter": filter}

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
        data = {
            "filter": filter,
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
