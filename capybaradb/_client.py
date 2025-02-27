import os
import requests
from capybaradb._database import Database


class CapybaraDB:
    """
    CapybaraDB - Main client class for interacting with CapybaraDB
    
    This is the entry point for the CapybaraDB Python SDK. It provides access to databases
    and collections within the CapybaraDB service.
    
    CapybaraDB is an AI-native database that combines NoSQL, vector storage, and object storage
    in a single platform. It allows you to store documents with embedded text fields that are
    automatically processed for semantic search.
    
    Usage:
        ```python
        from capybaradb import CapybaraDB, EmbText
        from dotenv import load_dotenv
        
        # Load environment variables (recommended for development)
        load_dotenv()
        
        # Initialize the client (requires CAPYBARA_PROJECT_ID and CAPYBARA_API_KEY env variables)
        client = CapybaraDB()
        
        # Access a database and collection
        db = client.db("my_database")
        collection = db.collection("my_collection")
        
        # Alternative syntax using attribute access
        collection = client.my_database.my_collection
        
        # Insert a document with an EmbText field (no manual embedding required)
        doc = {
            "title": "Sample Document",
            "content": EmbText("This is sample text that will be automatically embedded.")
        }
        collection.insert([doc])
        
        # Perform semantic search
        results = collection.query("sample text")
        ```
    
    Authentication:
        The SDK requires two environment variables:
        - CAPYBARA_PROJECT_ID: Your project ID from the CapybaraDB console
        - CAPYBARA_API_KEY: Your API key from the CapybaraDB console
        
        For production, these should be securely stored in your environment.
    """
    
    def __init__(self):
        """
        Creates a new CapybaraDB client instance.
        
        Automatically reads the project ID and API key from environment variables:
        - CAPYBARA_PROJECT_ID: Your project ID from the CapybaraDB console
        - CAPYBARA_API_KEY: Your API key from the CapybaraDB console
        
        Raises:
            ValueError: If either the project ID or API key is missing
        """
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
        
        This method provides access to a specific database within your CapybaraDB project.
        From the database, you can access collections and perform operations on documents.
        
        Args:
            db_name: The name of the database to access
            
        Returns:
            Database: A Database instance for the specified database
            
        Example:
            ```python
            db = client.db("my_database")
            ```
        """
        return Database(self.api_key, self.project_id, db_name)

    def __getattr__(self, name):
        """
        Dynamically return a 'Database' object when accessing as an attribute.
        
        This allows for a more intuitive syntax when accessing databases:
        ```python
        db = client.my_database  # Same as client.db("my_database")
        ```
        
        Args:
            name: The name of the database
            
        Returns:
            Database: Database instance for the specified database name
        """
        return self.db(name)

    def __getitem__(self, name):
        """
        Dynamically return a 'Database' object when accessing via dictionary syntax.
        
        This allows for an alternative syntax when accessing databases:
        ```python
        db = client["my_database"]  # Same as client.db("my_database")
        ```
        
        Args:
            name: The name of the database
            
        Returns:
            Database: Database instance for the specified database name
        """
        return self.db(name)
