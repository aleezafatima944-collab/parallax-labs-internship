
import json
from pathlib import Path

from datasets import load_dataset

NUM_DOCS = 6000
OUTPUT_PATH = Path("data/raw/wiki_raw.jsonl")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in dataset:
            record = {
                "id": row["id"],
                "title": row["title"],
                "text": row["text"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 500 == 0:
                print(f"Saved {count} documents...")
            if count >= NUM_DOCS:
                break

    print(f"Done. Saved {count} documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()