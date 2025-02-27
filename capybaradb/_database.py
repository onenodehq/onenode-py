from capybaradb._collection import Collection

class Database:
    """
    Database - Represents a database in CapybaraDB
    
    The Database class provides access to collections within a specific database in your CapybaraDB project.
    It serves as an intermediate layer between the CapybaraDB client and collections.
    
    In CapybaraDB, a database is a logical container for collections, similar to databases in other NoSQL systems.
    
    Usage:
        ```python
        from capybaradb import CapybaraDB
        
        client = CapybaraDB()
        
        # Get a database instance
        db = client.db("my_database")
        
        # Access a collection within the database
        collection = db.collection("my_collection")
        
        # Alternative syntax using attribute access
        collection = db.my_collection
        
        # Alternative syntax using dictionary access
        collection = db["my_collection"]
        ```
    """
    
    def __init__(self, api_key: str, project_id: str, db_name: str):
        """
        Creates a new Database instance.
        
        Note: You typically don't need to create this directly.
        Instead, use the `db()` method on a CapybaraDB client instance.
        
        Args:
            api_key: API key for authentication
            project_id: Project ID that identifies your CapybaraDB project
            db_name: Name of the database to access
        """
        self.api_key = api_key
        self.project_id = project_id
        self.db_name = db_name

    def collection(self, collection_name: str) -> Collection:
        """
        Get a collection instance within this database.
        
        This method provides access to a specific collection within the database.
        Collections in CapybaraDB are similar to collections in MongoDB or tables in SQL databases.
        They store documents (JSON objects) that can contain embedded text fields for semantic search.
        
        Args:
            collection_name: The name of the collection to access
            
        Returns:
            Collection: A Collection instance for the specified collection
            
        Example:
            ```python
            collection = db.collection("my_collection")
            
            # Insert documents
            collection.insert([{"name": "Document 1"}])
            
            # Query documents
            results = collection.find({"name": "Document 1"})
            ```
        """
        return Collection(self.api_key, self.project_id, self.db_name, collection_name)

    def __getattr__(self, name: str) -> Collection:
        """
        Dynamically return a 'Collection' object when accessing as an attribute.
        
        This allows for a more intuitive syntax when accessing collections:
        ```python
        collection = db.my_collection  # Same as db.collection("my_collection")
        ```
        
        Args:
            name: The name of the collection
            
        Returns:
            Collection: Collection instance for the specified collection name
        """
        return self.collection(name)

    def __getitem__(self, name: str) -> Collection:
        """
        Dynamically return a 'Collection' object when accessing via dictionary syntax.
        
        This allows for an alternative syntax when accessing collections:
        ```python
        collection = db["my_collection"]  # Same as db.collection("my_collection")
        ```
        
        Args:
            name: The name of the collection
            
        Returns:
            Collection: Collection instance for the specified collection name
        """
        return self.collection(name)
