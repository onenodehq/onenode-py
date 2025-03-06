# CapybaraDB Python SDK

> The official Python library for CapybaraDB - the chillest AI-native database.  
> **Store documents, vectors, and more — all in one place, with no need for extra vector DBs.**

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Sign Up and Get Credentials](#sign-up-and-get-credentials)
  - [Initialize Client](#initialize-client)
  - [Insert Documents (No Embedding Required!)](#insert-documents-no-embedding-required)
  - [Query Documents (Semantic Search)](#query-documents-semantic-search)
- [EmbJSON Data Types](#embjson-data-types)
  - [EmbText](#embtext)
    - [Basic Usage](#basic-usage)
    - [Customized Usage](#customized-usage)
    - [Parameter Reference](#parameter-reference)
    - [How It Works](#how-it-works)
    - [Accessing Generated Chunks](#accessing-generated-chunks)
    - [Usage in Nested Fields](#usage-in-nested-fields)
  - [EmbImage](#embimage)
    - [Basic Usage](#basic-usage-1)
    - [Customized Usage](#customized-usage-1)
    - [Parameter Reference](#parameter-reference-1)
    - [How It Works](#how-it-works-1)
    - [Querying Images](#querying-images)
- [License](#license)
- [Contact](#contact)

---

## Features

- **NoSQL + Vector + Object Storage** in one platform.  
- **No External Embedding Steps** — Just insert text with `EmbText`, CapybaraDB does the rest!  
- **Built-in Semantic Search** — Perform similarity-based queries without external services.  
- **Production-Ready** — Securely store your API key using environment variables.  

## Installation

```bash
pip install capybaradb
```

> **Note:** For local development, you can store your key in a `.env` file or assign it to a variable directly. Avoid hardcoding credentials in production.

---

## Quick Start

### Sign Up and Get Credentials

1. **Sign Up** at [CapybaraDB](https://capybaradb.co).  
2. Retrieve your **API Key** and **Project ID** from the developer console.  
3. **Store these securely** (e.g., in environment variables).

### Initialize Client

```python
import os
from capybaradb import CapybaraDB

# Load environment variables (for local development)
# In production, set these in your environment
os.environ["CAPYBARA_API_KEY"] = "your-api-key"
os.environ["CAPYBARA_PROJECT_ID"] = "your-project-id"

# Initialize the client
client = CapybaraDB()

# Access a database and collection
db = client.db("my_database")
collection = db.collection("my_collection")

# Alternative syntax using attribute access
collection = client.my_database.my_collection
```

---

### Insert Documents (No Embedding Required!)

```python
from capybaradb import CapybaraDB, EmbText

# Initialize the client
client = CapybaraDB()
collection = client.my_database.my_collection

# Define a document with an EmbText field
document = {
    "name": "Alice",
    "age": 7,
    "background": EmbText(
        "Through the Looking-Glass follows Alice as she steps into a fantastical world..."
    )
}

# Insert the document
result = collection.insert_one(document)
print(f"Inserted document with ID: {result.inserted_id}")
```

**What Happens Under the Hood?**  
- Text fields wrapped as `EmbText` are automatically chunked and embedded.  
- The resulting vectors are indexed for semantic queries.
- All processing happens asynchronously in the background.

---

### Query Documents (Semantic Search)

```python
from capybaradb import CapybaraDB

# Initialize the client
client = CapybaraDB()
collection = client.my_database.my_collection

# Simple text query
user_query = "What is the capital of France?"
filter_dict = {"category": "geography"} # Optional
projection = {"mode": "include", "fields": ["title", "content"]} # Optional

# Perform semantic search
response = collection.query(user_query, filter_dict, projection)
print("Query matches:", response.matches)

# Access the first match
if response.matches:
    match = response.matches[0]
    print(f"Matched chunk: {match.chunk}")
    print(f"Field path: {match.path}")
    print(f"Similarity score: {match.score}")
    print(f"Document ID: {match.document._id}")
```

**Example Response**:

```python
{
  "matches": [
    {
      "chunk": "Through the Looking-Glass follows Alice...",
      "path": "background",
      "score": 0.703643203,
      "document": {
        "_id": ObjectId("671bf91580bffb6387b4f3d2")
      }
    }
  ]
}
```

---

## EmbJSON Data Types

CapybaraDB extends JSON with AI-friendly data types like `EmbText`, making text embeddings and indexing automatic.  
No need for a separate vector DB or embedding service — CapybaraDB handles chunking, embedding, and indexing asynchronously.

### EmbText

`EmbText` is a specialized data type for storing and embedding text in CapybaraDB. It enables semantic search capabilities by automatically chunking, embedding, and indexing text.

When stored in the database, the text is processed asynchronously in the background:
1. The text is chunked based on the specified parameters
2. Each chunk is embedded using the specified embedding model
3. The embeddings are indexed for efficient semantic search

#### Basic Usage

Below is the simplest way to use `EmbText`:

```python
from capybaradb import EmbText

# Storing a single text field that you want to embed
document = {
  "field_name": EmbText("Alice is a data scientist with expertise in AI and machine learning. She has led several projects in natural language processing.")
}
```

This snippet creates an `EmbText` object containing the text. By default, it uses the `text-embedding-3-small` model and sensible defaults for chunking and overlap.

#### Customized Usage

If you have specific requirements (e.g., a different embedding model or particular chunking strategy), customize `EmbText` by specifying additional parameters:

```python
from capybaradb import EmbText, EmbModels

document = {
    "field_name": EmbText(
        text="Alice is a data scientist with expertise in AI and machine learning. She has led several projects in natural language processing.",
        emb_model=EmbModels.TEXT_EMBEDDING_3_LARGE,  # Change the default model
        max_chunk_size=200,                          # Configure chunk sizes
        chunk_overlap=20,                            # Overlap between chunks
        is_separator_regex=False,                    # Are separators plain strings or regex?
        separators=[
            "\n\n",
            "\n",
        ],
        keep_separator=False,                        # Keep or remove the separator in chunks
    )
}
```

#### Parameter Reference

| **Parameter**          | **Description**                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **text**               | The core content for `EmbText`. This text is automatically chunked and embedded for semantic search.                                              |
| **emb_model**          | Which embedding model to use. Defaults to `text-embedding-3-small`. You can choose from other supported models, such as `text-embedding-3-large`. |
| **max_chunk_size**     | Maximum character length of each chunk. Larger chunks reduce the total chunk count but may reduce search efficiency (due to bigger embeddings).   |
| **chunk_overlap**      | Overlapping character count between consecutive chunks, useful for preserving context at chunk boundaries.                                        |
| **is_separator_regex** | Whether to treat each separator in `separators` as a regular expression. Defaults to `False`.                                                     |
| **separators**         | A list of separator strings (or regex patterns) used to split the text. For instance, `["\n\n", "\n"]` can split paragraphs or single lines.      |
| **keep_separator**     | If `True`, separators remain in the chunked text. If `False`, they are stripped out.                                                              |
| **chunks**             | **Auto-generated by the database** after the text is processed. It is **not** set by the user, and is available only after embedding completes.   |

#### How It Works

Whenever you insert a document containing `EmbText` into CapybaraDB, three main steps happen **asynchronously**:

1. **Chunking**  
   The text is divided into chunks based on `max_chunk_size`, `chunk_overlap`, and any specified `separators`. This ensures the text is broken down into optimally sized segments.

2. **Embedding**  
   Each chunk is transformed into a vector representation using the specified `emb_model`. This step captures the semantic essence of the text.

3. **Indexing**  
   The embeddings are indexed for efficient semantic search. Because these steps occur in the background, you get immediate responses to your write operations, but actual query availability may lag slightly behind the write.

#### Accessing Generated Chunks

The `chunks` attribute is **auto-added** by the database after the text finishes embedding and indexing. For instance:

```python
# Assume this EmbText has been inserted and processed
emb_text = document["field_name"]  

print(emb_text.text)
# "Alice is a data scientist with expertise in AI and machine learning. She has led several projects in natural language processing."

print(emb_text.chunks)
# [
#   "Alice is a data scientist",
#   "with expertise in AI",
#   "and machine learning.",
#   "She has led several projects",
#   "in natural language processing."
# ]
```

#### Usage in Nested Fields

`EmbText` can be embedded anywhere in your document, including nested objects:

```python
document = {
  "profile": {
    "name": "Bob",
    "bio": EmbText(
      "Bob has over a decade of experience in AI, focusing on neural networks and deep learning."
    )
  }
}
```

### EmbImage

`EmbImage` is a specialized data type for storing and processing images in CapybaraDB. It enables multimodal capabilities by storing images that can be:

1. Processed by vision models to extract textual descriptions
2. Embedded for vector search (using the extracted descriptions)
3. Stored alongside other document data

When stored in the database, the image is processed asynchronously in the background:
- If a vision model is specified, the image is analyzed to generate textual descriptions
- If an embedding model is specified, these descriptions are embedded for semantic search
- The results are stored in the 'chunks' property

#### Basic Usage

Below is the simplest way to use `EmbImage`:

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

This snippet creates an `EmbImage` object containing your base64-encoded image data. By default, no specific models are set and all other parameters remain optional.

#### Customized Usage

If you have specific requirements (e.g., using a particular embedding or vision model), customize `EmbImage` by specifying additional parameters:

```python
from capybaradb import EmbImage, EmbModels, VisionModels
import base64

# Read an image file and convert to base64
with open("path/to/image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

document = {
    "title": "Product Image",
    "description": "Our latest product",
    "image": EmbImage(
        data=image_data,                                  # Base64-encoded image
        vision_model=VisionModels.GPT_4O,                 # Vision model for analysis
        emb_model=EmbModels.TEXT_EMBEDDING_3_SMALL,       # For embedding descriptions
        max_chunk_size=200,                               # Configure chunk sizes
        chunk_overlap=20,                                 # Overlap between chunks
        is_separator_regex=False,                         # Are separators plain strings or regex?
        separators=["\n\n", "\n"],                        # Separators for chunking
        keep_separator=False                              # Keep or remove separators
    )
}
```

#### Parameter Reference

| **Parameter**          | **Description**                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **data**               | The base64 encoded image data. This image is processed and embedded for semantic search.                                                          |
| **vision_model**       | Which vision model to use for processing the image. Defaults to `None`. Supported models include `GPT_4O_MINI`, `GPT_4O`, `GPT_4O_TURBO`, and `GPT_O1`. |
| **emb_model**          | Which embedding model to use for text chunks. Defaults to `None`. Supported models include `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002`. |
| **max_chunk_size**     | Maximum character length for each text chunk. Used when processing vision model output.                                                           |
| **chunk_overlap**      | Overlapping character count between consecutive chunks, useful for preserving context at chunk boundaries.                                        |
| **is_separator_regex** | Whether to treat each separator in `separators` as a regular expression. Defaults to `False`.                                                     |
| **separators**         | A list of separator strings (or regex patterns) used during processing. While more common in text, these may also apply to image metadata or descriptions if present. |
| **keep_separator**     | If `True`, separators remain in the processed data. If `False`, they are removed.                                                                 |
| **chunks**             | **Auto-generated by the database** after processing the image. It is **not** set by the user, and is available only after embedding completes.    |

#### How It Works

Whenever you insert a document containing `EmbImage` into CapybaraDB, the following steps occur **asynchronously**:

1. **Data Validation and Decoding**  
   The base64 image data is validated (ensuring it's properly encoded) and decoded as needed.

2. **Vision Model Processing (if specified)**  
   If a vision model is specified, the image is analyzed to generate textual descriptions.

3. **Embedding (if specified)**  
   If an embedding model is specified, the textual descriptions are transformed into vector representations.

4. **Indexing**  
   The resulting embeddings are indexed for efficient semantic search. These steps happen in the background, so while write operations are fast, query availability may have a slight delay.

#### Querying Images

Once the embedding and indexing steps are complete, your `EmbImage` fields become searchable. To perform semantic queries on image data, use the standard query operations:

```python
from capybaradb import CapybaraDB

# Initialize the client
client = CapybaraDB()
collection = client.my_database.my_collection

# Query for images with similar content
results = collection.query("product with blue background")

# Access the first match
if results.matches:
    match = results.matches[0]
    print(f"Matched chunk: {match.chunk}")
    print(f"Field path: {match.path}")
    print(f"Similarity score: {match.score}")
    print(f"Document ID: {match.document._id}")
```

## License

[Apache 2.0](LICENSE) © 2025 CapybaraDB

## Contact

- **Questions?** [Email us](mailto:hello@capybaradb.co)  
- **Website:** [capybaradb.co](https://capybaradb.co)

Happy coding with CapybaraDB!
