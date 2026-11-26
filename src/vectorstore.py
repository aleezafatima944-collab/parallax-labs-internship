"""ChromaDB helpers: a persistent local collection that uses cosine distance."""
from pathlib import Path

import chromadb

DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "wiki_chunks"


def get_client(path=DB_PATH):
    """Open (or create) a persistent ChromaDB database on disk."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(client=None, name=COLLECTION_NAME):
    """Get the chunk collection, creating it with cosine distance if needed."""
    client = client or get_client()
    return client.get_or_create_collection(
        name=name, configuration={"hnsw": {"space": "cosine"}}
    )
    