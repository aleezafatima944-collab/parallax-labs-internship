
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import clean_text

RAW_PATH = Path("data/raw/wiki_raw.jsonl")
OUTPUT_PATH = Path("data/processed/clean_corpus.parquet")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    total = 0
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total += 1
            cleaned = clean_text(record["text"])
            if cleaned is None:
                continue
            kept.append(
                {
                    "id": record["id"],
                    "title": record["title"],
                    "text": cleaned,
                }
            )
            if total % 500 == 0:
                print(f"Processed {total} documents...")

    df = pd.DataFrame(kept)
    df.to_parquet(OUTPUT_PATH, index=False)

    dropped = total - len(kept)
    print(f"Total raw documents: {total}")
    print(f"Kept after cleaning: {len(kept)}")
    print(f"Dropped (empty or non-English): {dropped}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()