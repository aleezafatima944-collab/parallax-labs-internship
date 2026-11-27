"""Search the ChromaDB collection from the command line.

Usage:
    python scripts/search_demo.py "Who was the 16th president of the United States?"
    python scripts/search_demo.py "what is anarchism" --top-k 3
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import load_model
from src.search import search
from src.vectorstore import get_collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="your question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    query = " ".join(args.query)

    collection = get_collection()
    if collection.count() == 0:
        print("The collection is empty. Run scripts/ingest_chroma.py first.")
        return

    model = load_model()
    start = time.perf_counter()
    hits = search(collection, model, query, n_results=args.top_k)
    milliseconds = (time.perf_counter() - start) * 1000

    print(f"Query: {query}")
    print(f"Searched {collection.count()} chunks in {milliseconds:.0f} ms\n")
    for rank, hit in enumerate(hits, start=1):
        snippet = hit["text"][:200].replace("\n", " ")
        print(f"{rank}. {hit['title']} (chunk {hit['chunk_index']}) | distance {hit['distance']:.3f}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()