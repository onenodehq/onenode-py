import os
import requests
from capybaradb.database import Database


class CapybaraDB:
    def __init__(self):
        # Ensure that environment variables are checked and valid
        self.project_id = os.getenv("CAPYBARA_PROJECT_ID", "")
        self.api_key = os.getenv("CAPYBARA_API_KEY", "")

        # Validate that both values are provided
        if not self.project_id:
            raise ValueError(
                "Project ID must be specified either as an argument or in the environment variable CAPYBARA_PROJECT_ID."
            )

        if not self.api_key:
            raise ValueError(
                "API Key must be specified either as an argument or in the environment variable CAPYBARA_API_KEY."
            )

        self.base_url = f"https://api.capybaradb.co/{self.project_id}".rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

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
