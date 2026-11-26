"""Embed chunks and store them in a persistent ChromaDB collection.

Safe to stop and re-run: chunks that are already stored are skipped.

Usage:
    python scripts/ingest_chroma.py --max-docs 1000
    python scripts/ingest_chroma.py            (all documents)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import embed_texts, load_model, memory_mb
from src.vectorstore import get_collection

CHUNKS_PATH = Path("data/processed/chunks.parquet")
BATCH_SIZE = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None,
                        help="only ingest the first N documents")
    args = parser.parse_args()

    chunks = pd.read_parquet(CHUNKS_PATH)
    if args.max_docs:
        keep = chunks["doc_id"].drop_duplicates().head(args.max_docs)
        chunks = chunks[chunks["doc_id"].isin(keep)]

    collection = get_collection()
    existing = set(collection.get(include=[])["ids"])
    todo = chunks[~chunks["chunk_id"].isin(existing)]
    print(f"Chunks selected: {len(chunks)} | already stored: {len(existing)} "
          f"| to do now: {len(todo)}")
    if todo.empty:
        print("Nothing to do.")
        return

    model = load_model()
    start = time.perf_counter()
    done = 0
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[i : i + BATCH_SIZE]
        embeddings, _ = embed_texts(model, batch["text"].tolist())
        collection.add(
            ids=batch["chunk_id"].tolist(),
            embeddings=embeddings,
            documents=batch["text"].tolist(),
            metadatas=[
                {"doc_id": d, "title": t, "chunk_index": int(n)}
                for d, t, n in zip(batch["doc_id"], batch["title"], batch["chunk_index"])
            ],
        )
        done += len(batch)
        elapsed = time.perf_counter() - start
        rate = done / elapsed
        eta = (len(todo) - done) / rate / 60
        print(f"{done}/{len(todo)} chunks | {rate:.1f} chunks/s | "
              f"memory {memory_mb():.0f} MB | about {eta:.0f} min left")

    total = time.perf_counter() - start
    print(f"Finished in {total / 60:.1f} minutes. Collection now holds {collection.count()} chunks.")


if __name__ == "__main__":
    main()