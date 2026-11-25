"""Embedding helpers that log latency and memory use."""
import time

import psutil

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def memory_mb():
    """Memory used by this Python process, in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def load_model(model_name=MODEL_NAME):
    """Load the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(model, texts, batch_size=64):
    """Embed texts. Returns (embeddings, stats) with timing and memory info."""
    memory_before = memory_mb()
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    seconds = time.perf_counter() - start
    stats = {
        "num_texts": len(texts),
        "seconds": seconds,
        "ms_per_text": 1000 * seconds / max(len(texts), 1),
        "texts_per_second": len(texts) / seconds if seconds > 0 else float("inf"),
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_mb(),
    }
    return embeddings, stats