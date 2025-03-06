"""
CapybaraDB Python SDK

The official Python library for CapybaraDB - an AI-native database that combines
NoSQL, vector storage, and object storage in a single platform.

This package provides a simple, intuitive interface for:
- Storing documents with embedded text fields (no manual embedding required)
- Performing semantic searches on your data
- Managing collections and databases

Key components:
- CapybaraDB: Main client class for connecting to the service
- EmbText: Special data type for text that will be automatically embedded
- EmbModels: Constants for supported embedding models
- EmbImage: Special data type for images that can be processed by vision models
- VisionModels: Constants for supported vision models

Important Note:
CapybaraDB processes embeddings asynchronously on the server side. When you insert documents
with EmbText or EmbImage fields, there will be a short delay (typically a few seconds) before
these documents become available for semantic search. This is because the embedding generation
happens in the background after the document is stored.

Basic usage:
```python
from capybaradb import CapybaraDB, EmbText
from dotenv import load_dotenv

# Load environment variables (CAPYBARA_API_KEY and CAPYBARA_PROJECT_ID)
load_dotenv()

# Initialize the client
client = CapybaraDB()

# Access a database and collection
collection = client.my_database.my_collection

# Insert a document with embedded text
doc = {
    "title": "Sample Document",
    "content": EmbText("This text will be automatically embedded for semantic search")
}
collection.insert([doc])

# Note: There will be a short delay before the document is available for semantic search
# as embeddings are processed asynchronously on the server side

# Perform semantic search (after embeddings have been processed)
results = collection.query("semantic search")
```

EmbText Usage:
--------------
EmbText is a specialized data type for storing and embedding text in CapybaraDB. It enables 
semantic search capabilities by automatically chunking, embedding, and indexing text.

Basic Usage:
```python
from capybaradb import EmbText

# Storing a single text field that you want to embed
document = {
  "field_name": EmbText("This text will be automatically embedded for semantic search")
}
```

Customized Usage:
```python
from capybaradb import EmbText, EmbModels

document = {
    "field_name": EmbText(
        text="This text will be automatically embedded for semantic search",
        emb_model=EmbModels.TEXT_EMBEDDING_3_LARGE,  # Change the default model
        max_chunk_size=200,                          # Configure chunk sizes
        chunk_overlap=20,                            # Overlap between chunks
        is_separator_regex=False,                    # Are separators plain strings or regex?
        separators=["\n\n", "\n"],                   # Separators for chunking
        keep_separator=False                         # Keep or remove separators
    )
}
```

EmbImage Usage:
---------------
EmbImage is a specialized data type for storing and processing images in CapybaraDB. It enables 
multimodal capabilities by storing images that can be processed by vision models and embedded 
for semantic search.

Basic Usage:
```python
from capybaradb import EmbImage
import base64

# Read an image file and convert to base64
with open("path/to/image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Storing a single image field
document = {
    "title": "Product Image",
    "image": EmbImage(image_data)
}
```

Customized Usage:
```python
from capybaradb import EmbImage, EmbModels, VisionModels
import base64

# Read an image file and convert to base64
with open("path/to/image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

document = {
    "title": "Product Image",
    "image": EmbImage(
        data=image_data,                                  # Base64-encoded image
        vision_model=VisionModels.GPT_4O,                 # Vision model for analysis
        emb_model=EmbModels.TEXT_EMBEDDING_3_SMALL,       # For embedding descriptions
        max_chunk_size=200,                               # Configure chunk sizes
        chunk_overlap=20                                  # Overlap between chunks
    )
}
```

How It Works:
When you insert a document with EmbText or EmbImage fields:
1. The data is stored immediately in the database
2. Asynchronously, the text/image is processed:
   - For EmbText: The text is chunked and embedded
   - For EmbImage: The image is analyzed by the vision model (if specified) and embedded
3. The resulting embeddings are indexed for semantic search
4. The chunks are stored in the document for future reference

For more information, see the documentation at https://capybaradb.co/docs
"""

from ._client import CapybaraDB
from ._emb_json._emb_text import EmbText
from ._emb_json._emb_models import EmbModels
from ._emb_json._emb_image import EmbImage
from ._emb_json._vision_models import VisionModels
import bson

__all__ = ["CapybaraDB", "EmbText", "EmbModels", "EmbImage", "VisionModels", "bson"]
