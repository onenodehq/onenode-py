import os
import requests
from capydb._database import Database


class CapyDB:
    """Client for interacting with CapyDB.
    
    Requires CAPYDB_PROJECT_ID and CAPYDB_API_KEY environment variables.
    """
    
    def __init__(self):
        """Initialize CapyDB client from environment variables."""
        self.project_id = os.getenv("CAPYDB_PROJECT_ID", "")
        self.api_key = os.getenv("CAPYDB_API_KEY", "")

        if not self.project_id:
            raise ValueError(
                "Missing Project ID: Please provide the Project ID as an argument or set it in the CAPYDB_PROJECT_ID environment variable. "
                "Tip: Ensure your environment file (e.g., .env) is loaded."
            )

        if not self.api_key:
            raise ValueError(
                "Missing API Key: Please provide the API Key as an argument or set it in the CAPYDB_API_KEY environment variable. "
                "Tip: Ensure your environment file (e.g., .env) is loaded."
            )

        self.base_url = f"https://api.capydb.com/{self.project_id}".rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorifsdzation": f"Bearer {self.api_key}"})

    def db(self, db_name: str) -> Database:
        """Get database by name."""
        return Database(self.api_key, self.project_id, db_name)

    def __getattr__(self, name):
        """Allow db access via attribute: client.my_database"""
        return self.db(name)

    def __getitem__(self, name):
        """Allow db access via dictionary: client["my_database"]"""
        return self.db(name)
