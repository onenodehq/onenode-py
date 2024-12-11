import requests

class Collection:
    def __init__(self, api_key: str, project_id: str, db_name: str, collection_name: str):
        self.api_key = api_key
        self.project_id = project_id
        self.db_name = db_name
        self.collection_name = collection_name

    def get_collection_url(self) -> str:
        return f"https://api.onenode.ai/v1/db/{self.project_id}_{self.db_name}/collection/{self.collection_name}/document"

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def transform_emb_text(self, document: dict) -> dict:
        for key, value in document.items():
            if isinstance(value, dict):
                document[key] = self.transform_emb_text(value)
        return document

    def insert_one(self, document: dict) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_document = self.transform_emb_text(document)
        data = {"documents": [transformed_document]}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def insert_many(self, documents: list[dict]) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_documents = [self.transform_emb_text(doc) for doc in documents]
        data = {"documents": transformed_documents}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def update(self, filter: dict, update: dict, upsert: bool = False) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        transformed_update = self.transform_emb_text(update)
        data = {"filter": filter, "update": transformed_update, "upsert": upsert}

        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def delete(self, filter: dict) -> dict:
        url = self.get_collection_url()
        headers = self.get_headers()
        data = {"filter": filter}

        response = requests.delete(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def find(self, filter: dict, projection: dict = None, sort: dict = None, limit: int = None, skip: int = None) -> list[dict]:
        url = f"{self.get_collection_url()}/find"
        headers = self.get_headers()
        data = {"filter": filter, "projection": projection, "sort": sort, "limit": limit, "skip": skip}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def query(self, query: str, emb_model: str, top_k: int, include_values: bool = False, projection: dict = None) -> list[dict]:
        url = f"{self.get_collection_url()}/query"
        headers = self.get_headers()
        data = {
            "query": query,
            "emb_model": emb_model,
            "top_k": top_k,
            "include_values": include_values,
            "projection": projection,
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
