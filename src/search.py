"""Semantic similarity search over the ChromaDB chunk collection."""


def search(collection, model, query, n_results=5):
    """Return the n_results chunks closest in meaning to the query.

    Each hit is a dict with chunk_id, doc_id, title, chunk_index, distance and text.
    A smaller cosine distance means a more similar chunk.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if collection.count() == 0:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "chunk_index": meta["chunk_index"],
                "distance": distance,
                "text": text,
            }
        )
    return hits