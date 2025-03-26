import os
import requests
from capybaradb._database import Database


class CapybaraDB:
    """Client for interacting with CapybaraDB.
    
    Requires CAPYBARA_PROJECT_ID and CAPYBARA_API_KEY environment variables.
    """
    
    def __init__(self):
        """Initialize CapybaraDB client from environment variables."""
        self.project_id = os.getenv("CAPYBARA_PROJECT_ID", "")
        self.api_key = os.getenv("CAPYBARA_API_KEY", "")

        if not self.project_id:
            raise ValueError(
                "Missing Project ID: Please provide the Project ID as an argument or set it in the CAPYBARA_PROJECT_ID environment variable. "
                "Tip: Ensure your environment file (e.g., .env) is loaded."
            )

        if not self.api_key:
            raise ValueError(
                "Missing API Key: Please provide the API Key as an argument or set it in the CAPYBARA_API_KEY environment variable. "
                "Tip: Ensure your environment file (e.g., .env) is loaded."
            )

        self.base_url = f"https://api.capybaradb.co/{self.project_id}".rstrip("/")
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
