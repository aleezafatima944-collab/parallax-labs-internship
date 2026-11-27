import numpy as np
import pytest

from src.search import search
from src.vectorstore import get_client, get_collection

VOCAB = ["cat", "dog", "pet", "stock", "market", "money", "river", "water", "boat"]


class FakeModel:
    """Tiny stand-in for the real model: counts known words, then normalizes."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.array([words.count(w) for w in VOCAB], dtype="float32") + 0.01
            vectors.append(vec / np.linalg.norm(vec))
        return np.array(vectors)


def make_collection(tmp_path, texts):
    collection = get_collection(get_client(tmp_path / "db"), name="test_chunks")
    if texts:
        collection.add(
            ids=[f"d{i}_0" for i in range(len(texts))],
            embeddings=FakeModel().encode(texts),
            documents=texts,
            metadatas=[
                {"doc_id": f"d{i}", "title": f"Doc {i}", "chunk_index": 0}
                for i in range(len(texts))
            ],
        )
    return collection


TEXTS = ["cat dog pet", "stock market money", "river water boat"]


def test_returns_most_similar_chunk_first(tmp_path):
    collection = make_collection(tmp_path, TEXTS)
    hits = search(collection, FakeModel(), "my pet cat", n_results=3)
    assert hits[0]["text"] == "cat dog pet"
    assert hits[0]["title"] == "Doc 0"


def test_distances_are_sorted_smallest_first(tmp_path):
    collection = make_collection(tmp_path, TEXTS)
    hits = search(collection, FakeModel(), "stock market", n_results=3)
    distances = [h["distance"] for h in hits]
    assert distances == sorted(distances)
    assert hits[0]["text"] == "stock market money"


def test_hit_has_expected_fields(tmp_path):
    collection = make_collection(tmp_path, TEXTS)
    hit = search(collection, FakeModel(), "boat", n_results=1)[0]
    assert set(hit) == {"chunk_id", "doc_id", "title", "chunk_index", "distance", "text"}


def test_asking_for_more_results_than_stored(tmp_path):
    collection = make_collection(tmp_path, TEXTS)
    assert len(search(collection, FakeModel(), "water", n_results=10)) == 3


def test_empty_collection_returns_no_hits(tmp_path):
    collection = make_collection(tmp_path, [])
    assert search(collection, FakeModel(), "anything") == []


def test_empty_query_raises_error(tmp_path):
    collection = make_collection(tmp_path, TEXTS)
    with pytest.raises(ValueError):
        search(collection, FakeModel(), "   ")