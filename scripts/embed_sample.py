"""Embed a sample of chunks to measure speed and memory on this machine."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import MODEL_NAME, embed_texts, load_model, memory_mb

CHUNKS_PATH = Path("data/processed/chunks.parquet")
SAMPLE_SIZE = 500


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    sample = chunks["text"].sample(min(SAMPLE_SIZE, len(chunks)), random_state=0).tolist()

    print(f"Memory before loading model: {memory_mb():.0f} MB")
    start = time.perf_counter()
    model = load_model()
    load_seconds = time.perf_counter() - start
    print(f"Model {MODEL_NAME} loaded in {load_seconds:.1f} seconds on {model.device}")
    print(f"Memory after loading model: {memory_mb():.0f} MB")

    model.encode(sample[:16], show_progress_bar=False)  # warm-up, not timed

    embeddings, stats = embed_texts(model, sample)

    print(f"Embedded {stats['num_texts']} chunks in {stats['seconds']:.1f} seconds")
    print(
        f"Speed: {stats['texts_per_second']:.1f} chunks/second "
        f"({stats['ms_per_text']:.1f} ms per chunk)"
    )
    print(f"Memory after embedding: {stats['memory_after_mb']:.0f} MB")
    print(f"Embedding shape: {embeddings.shape}")
    minutes = len(chunks) / stats["texts_per_second"] / 60
    print(f"Estimated time for all {len(chunks)} chunks: {minutes:.0f} minutes")


if __name__ == "__main__":
    main()