"""
CapybaraDB Python SDK

Official Python library for CapybaraDB - AI-native database with NoSQL, vector and object storage.

Basic usage:
```python
from capybaradb import CapybaraDB, EmbText
from dotenv import load_dotenv

load_dotenv()
client = CapybaraDB()
collection = client.my_database.my_collection
doc = {"title": "Sample", "content": EmbText("Text for embedding")}
collection.insert([doc])
results = collection.query("search query")
```

Docs: https://capybaradb.co/docs
"""

from ._client import CapybaraDB
from ._emb_json._emb_text import EmbText
from ._emb_json._emb_models import EmbModels
from ._emb_json._emb_image import EmbImage
from ._emb_json._vision_models import VisionModels
import bson

__all__ = ["CapybaraDB", "EmbText", "EmbModels", "EmbImage", "VisionModels", "bson"]
