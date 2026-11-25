"""Split the clean corpus into chunks and save them.

Reads data/processed/clean_corpus.parquet, chunks every document,
and writes data/processed/chunks.parquet.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents
from src.metrics import count_tokens, get_tokenizer

INPUT_PATH = Path("data/processed/clean_corpus.parquet")
OUTPUT_PATH = Path("data/processed/chunks.parquet")
TOKEN_SAMPLE = 2000


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Chunking {len(df)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    start = time.perf_counter()
    records = chunk_documents(df)
    elapsed = time.perf_counter() - start

    chunks = pd.DataFrame(records)
    chunks["char_count"] = chunks["text"].str.len()
    chunks.to_parquet(OUTPUT_PATH, index=False)

    sample = chunks["text"].sample(min(TOKEN_SAMPLE, len(chunks)), random_state=0).tolist()
    tokens = pd.Series(count_tokens(sample, get_tokenizer()))

    print(f"Chunking took {elapsed:.1f} seconds")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunks per document: {len(chunks) / len(df):.1f}")
    print(f"Average chunk length: {chunks['char_count'].mean():.0f} characters")
    print(f"Longest chunk: {chunks['char_count'].max()} characters")
    print(f"Tokens per chunk (sample of {len(sample)}): "
          f"average {tokens.mean():.0f}, max {tokens.max()}, "
          f"over 256: {(tokens > 256).sum()}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()