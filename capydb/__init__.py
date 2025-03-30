"""
CapyDB Python SDK

Official Python library for CapyDB - AI-native database with NoSQL, vector and object storage.

Basic usage:
```python
from capydb import CapyDB, EmbText
from dotenv import load_dotenv

load_dotenv()
client = CapyDB()
collection = client.my_database.my_collection
doc = {"title": "Sample", "content": EmbText("Text for embedding")}
collection.insert([doc])
results = collection.query("search query")
```

Docs: https://capydb.com/docs
"""

from ._client import CapyDB
from ._emb_json._emb_text import EmbText
from ._emb_json._emb_models import EmbModels
from ._emb_json._emb_image import EmbImage
from ._emb_json._vision_models import VisionModels
import bson

__all__ = ["CapyDB", "EmbText", "EmbModels", "EmbImage", "VisionModels", "bson"]
