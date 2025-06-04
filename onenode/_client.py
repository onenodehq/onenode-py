import os
import requests
from bson import ObjectId
from onenode._database import Database


class OneNode:
    """Client for interacting with OneNode.
    
    Requires ONENODE_PROJECT_ID and ONENODE_API_KEY environment variables,
    or will operate in anonymous mode if API key is not provided.
    """
    
    def __init__(self):
        """Initialize OneNode client from environment variables."""
        self.project_id = os.getenv("ONENODE_PROJECT_ID", "")
        self.api_key = os.getenv("ONENODE_API_KEY", "")
        self.is_anonymous = False

        # If no API key provided, enter anonymous mode
        if not self.api_key:
            self.is_anonymous = True
            # Generate or load anonymous project ID
            self.project_id = self._get_or_create_anonymous_project_id()
        else:
            # Authenticated mode - require project ID
            if not self.project_id:
                raise ValueError(
                    "Missing Project ID: Please provide the Project ID as an argument or set it in the ONENODE_PROJECT_ID environment variable. "
                    "Tip: Ensure your environment file (e.g., .env) is loaded."
                )

        self.base_url = f"https://api.onenode.ai/{self.project_id}".rstrip("/")
        self.session = requests.Session()
        
        # Only set authorization header if not in anonymous mode
        if not self.is_anonymous:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get_or_create_anonymous_project_id(self) -> str:
        """Get existing anonymous project ID from file or create a new one."""
        anon_file_path = ".onenode"
        
        # Try to load existing project ID
        if os.path.exists(anon_file_path):
            try:
                with open(anon_file_path, 'r') as f:
                    project_id = f.read().strip()
                    # Validate that it's a valid ObjectId format
                    ObjectId(project_id)
                    return project_id
            except (IOError, ValueError):
                # File exists but is invalid, will create new one
                pass
        
        # Generate new project ID
        new_project_id = str(ObjectId())
        
        # Save to file
        try:
            with open(anon_file_path, 'w') as f:
                f.write(new_project_id)
        except IOError:
            # If we can't write the file, just use the generated ID without persistence
            pass
            
        return new_project_id

    def db(self, db_name: str) -> Database:
        """Get database by name."""
        return Database(self.api_key, self.project_id, db_name, self.is_anonymous)

    def __getattr__(self, name):
        """Allow db access via attribute: client.my_database"""
        return self.db(name)

    def __getitem__(self, name):
        """Allow db access via dictionary: client["my_database"]"""
        return self.db(name)
