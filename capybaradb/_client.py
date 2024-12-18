import os
import requests
from capybaradb._database import Database


class CapybaraDB:
    def __init__(self):
        # Ensure that environment variables are checked and valid
        self.project_id = os.getenv("CAPYBARA_PROJECT_ID", "")
        self.api_key = os.getenv("CAPYBARA_API_KEY", "")

        # Validate that both values are provided
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
        """
        Get a database instance.
        :param db_name: The name of the database
        :return: Database instance
        """
        return Database(self.api_key, self.project_id, db_name)

    def __getattr__(self, name):
        """
        Dynamically return a 'Database' object when accessing as an attribute.
        :param name: The name of the database
        :return: Database instance
        """
        return self.db(name)

    def __getitem__(self, name):
        """
        Dynamically return a 'Database' object when accessing via dictionary syntax.
        :param name: The name of the database
        :return: Database instance
        """
        return self.db(name)
