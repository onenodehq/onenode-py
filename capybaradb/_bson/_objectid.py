from typing import Dict
import re


class ObjectId:
    """
    A class to represent MongoDB ObjectId in Python with MongoDB Extended JSON support.
    """

    OBJECT_ID_REGEX = re.compile(r"^[a-fA-F0-9]{24}$")

    def __init__(self, object_id: str):
        if not self.is_valid_object_id(object_id):
            raise ValueError(
                "Invalid ObjectId: must be a 24-character hexadecimal string."
            )
        self.object_id = object_id

    @staticmethod
    def is_valid_object_id(object_id: str) -> bool:
        """
        Validate that the given object_id is a 24-character hexadecimal string.
        """
        return isinstance(object_id, str) and bool(
            ObjectId.OBJECT_ID_REGEX.match(object_id)
        )

    def to_json(self) -> Dict[str, str]:
        """
        Convert the ObjectId instance to MongoDB Extended JSON format.
        """
        return {"$oid": self.object_id}

    @classmethod
    def from_json(cls, data: Dict[str, str]) -> "ObjectId":
        """
        Create an ObjectId instance from MongoDB Extended JSON format.
        """
        object_id = data.get("$oid")
        if not object_id:
            raise ValueError("Invalid Extended JSON for ObjectId: Missing '$oid' key.")
        return cls(object_id)

    def __repr__(self):
        return f"ObjectId('{self.object_id}')"
