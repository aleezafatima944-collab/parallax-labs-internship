import pandas as pd
import pytest

from src.chunking import chunk_documents, chunk_text

SENTENCES = " ".join(f"Sentence number {i} is here." for i in range(200))


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []
    assert chunk_text(None) == []


def test_short_text_stays_one_chunk():
    assert chunk_text("Hello world.") == ["Hello world."]


def test_chunks_never_exceed_chunk_size():
    chunks = chunk_text(SENTENCES, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(SENTENCES, chunk_size=200, overlap=40)
    for previous, following in zip(chunks, chunks[1:]):
        assert following[:15] in previous


def test_no_sentence_is_lost():
    chunks = chunk_text(SENTENCES, chunk_size=200, overlap=40)
    joined = " ".join(chunks)
    for i in range(200):
        assert f"Sentence number {i} is here." in joined


def test_long_word_without_spaces_is_split():
    chunks = chunk_text("a" * 1000, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_prefers_paragraph_breaks():
    text = ("A" * 150) + "\n\n" + ("B" * 150)
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert chunks[0] == "A" * 150
    assert chunks[1].endswith("B" * 150)


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=10)


def test_chunk_documents_records():
    df = pd.DataFrame(
        {"id": ["7", "8"], "title": ["First", "Second"], "text": [SENTENCES, "Tiny doc."]}
    )
    records = chunk_documents(df, chunk_size=200, overlap=40)
    first = [r for r in records if r["doc_id"] == "7"]
    assert [r["chunk_index"] for r in first] == list(range(len(first)))
    assert first[0]["chunk_id"] == "7_0"
    assert first[0]["title"] == "First"
    second = [r for r in records if r["doc_id"] == "8"]
    assert len(second) == 1 and second[0]["chunk_id"] == "8_0"