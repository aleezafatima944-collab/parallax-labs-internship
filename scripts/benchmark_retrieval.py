"""Benchmark retrieval latency and check nearest-neighbor results.

For 10 sample queries this script measures how long embedding the question and
searching ChromaDB take, checks that the expected article is among the top
results, and compares ChromaDB's answers with an exact brute-force search.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import MODEL_NAME, load_model, memory_mb
from src.vectorstore import get_collection

TOP_K = 5
REPEATS = 5
RESULTS_PATH = Path("results/benchmark_results.json")

# (question, title of the article that should be found)
QUERIES = [
    ("Who was the French sculptor known for The Thinker?", "Auguste Rodin"),
    ("What is the largest artery in the human body?", "Aorta"),
    ("What is the Arabic script and how is it written?", "Arabic alphabet"),
    ("Which philosopher wrote Language, Truth and Logic?", "A. J. Ayer"),
    ("Who founded the Bavarian Illuminati?", "Adam Weishaupt"),
    ("What is the Accelerated Graphics Port interface used for?", "Accelerated Graphics Port"),
    ("Which American football team plays in Arizona?", "Arizona Cardinals"),
    ("What is Area 51 and where is it located?", "Area 51"),
    ("How do plants respond to drought and salt stress?", "Abiotic stress"),
    ("Which astronomer tested general relativity during the 1919 solar eclipse?", "Arthur Eddington"),
]


def timed(function):
    """Run function() and return (result, milliseconds)."""
    start = time.perf_counter()
    value = function()
    return value, (time.perf_counter() - start) * 1000


def main():
    collection = get_collection()
    total_chunks = collection.count()
    if total_chunks == 0:
        print("The collection is empty. Run scripts/ingest_chroma.py first.")
        return

    model = load_model()
    model.encode(["warm up"], show_progress_bar=False)  # not timed

    stored = collection.get(include=["embeddings"])
    stored_ids = np.array(stored["ids"])
    stored_vectors = np.asarray(stored["embeddings"], dtype="float32")

    rows = []
    for question, expected in QUERIES:
        embed_times, search_times = [], []
        for _ in range(REPEATS):
            vector, embed_ms = timed(lambda: model.encode(
                [question], normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True))
            result, search_ms = timed(lambda: collection.query(
                query_embeddings=vector, n_results=TOP_K,
                include=["metadatas", "distances"]))
            embed_times.append(embed_ms)
            search_times.append(search_ms)

        titles = [m["title"] for m in result["metadatas"][0]]
        rank = titles.index(expected) + 1 if expected in titles else None

        exact_ids = stored_ids[np.argsort(1 - stored_vectors @ vector[0])[:TOP_K]]
        agreement = len(set(exact_ids) & set(result["ids"][0])) / TOP_K

        rows.append({
            "question": question,
            "expected_title": expected,
            "rank_in_top_k": rank,
            "top_result": titles[0],
            "best_distance": round(result["distances"][0][0], 3),
            "embed_ms": round(statistics.median(embed_times), 1),
            "search_ms": round(statistics.median(search_times), 1),
            "exact_agreement": agreement,
        })

    print(f"\n{'#':<3}{'Expected article':<28}{'Rank':<6}{'Top result':<28}{'Embed ms':<10}{'Search ms':<10}")
    for i, r in enumerate(rows, start=1):
        rank = r["rank_in_top_k"] if r["rank_in_top_k"] else "miss"
        print(f"{i:<3}{r['expected_title'][:26]:<28}{str(rank):<6}"
              f"{r['top_result'][:26]:<28}{r['embed_ms']:<10}{r['search_ms']:<10}")

    totals = [r["embed_ms"] + r["search_ms"] for r in rows]
    summary = {
        "model": MODEL_NAME,
        "chunks_in_collection": total_chunks,
        "top_k": TOP_K,
        "repeats_per_query": REPEATS,
        "hit_at_1": sum(r["rank_in_top_k"] == 1 for r in rows),
        "hit_at_5": sum(r["rank_in_top_k"] is not None for r in rows),
        "queries": len(rows),
        "mean_embed_ms": round(statistics.mean(r["embed_ms"] for r in rows), 1),
        "mean_search_ms": round(statistics.mean(r["search_ms"] for r in rows), 1),
        "mean_total_ms": round(statistics.mean(totals), 1),
        "slowest_total_ms": round(max(totals), 1),
        "exact_search_agreement": round(statistics.mean(r["exact_agreement"] for r in rows), 3),
        "memory_mb": round(memory_mb()),
    }
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()