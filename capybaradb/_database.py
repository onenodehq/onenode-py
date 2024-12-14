from capybaradb._collection import Collection

class Database:
    def __init__(self, api_key: str, project_id: str, db_name: str):
        self.api_key = api_key
        self.project_id = project_id
        self.db_name = db_name

    def collection(self, collection_name: str) -> Collection:
        """
        Get a collection instance within this database.
        :param collection_name: The name of the collection
        :return: Collection instance
        """
        return Collection(self.api_key, self.project_id, self.db_name, collection_name)

    def __getattr__(self, name: str) -> Collection:
        """
        Dynamically return a 'Collection' object when accessing as an attribute.
        :param name: The name of the collection
        :return: Collection instance
        """
        return self.collection(name)

    def __getitem__(self, name: str) -> Collection:
        """
        Dynamically return a 'Collection' object when accessing via dictionary syntax.
        :param name: The name of the collection
        :return: Collection instance
        """
        return self.collection(name)
