# Parallax Labs Internship

Hi, I'm Aleeza. This is the single repository for my whole Parallax Labs internship. Every week I'll add my new work here and update this README with what changed, what got installed, and how to run it.

---

## Week 1: Environment, Data Ingestion & Preprocessing

### What this week was about

Before you can build anything smart with text, you need three boring-but-important things: a setup that works the same on every machine, a real pile of text to work with, and a way to make that text clean and consistent. That's what I built this week.

By the end, I had a verified environment, 6,000 real Wikipedia articles, a small tested cleaning pipeline, and a clean dataset of **5,982 documents** saved as `data/processed/clean_corpus.parquet`.

### What's in the repo

- `scripts/verify_env.py` checks that the core libraries import (pandas, sentence-transformers, ChromaDB, spaCy), that the spaCy English model loads, and whether a CUDA GPU is available.
- `scripts/download_data.py` streams 6,000 English Wikipedia articles into `data/raw/wiki_raw.jsonl`.
- `src/preprocessing.py` holds the cleaning functions.
- `tests/test_preprocessing.py` has 12 unit tests for those functions.
- `scripts/run_preprocessing.py` runs the cleaning on the whole corpus and writes the parquet file.

### How I approached it

**Picking the data.** The task allowed ArXiv, Wikipedia or Reddit. I went with Wikipedia (the `wikimedia/wikipedia` dataset on Hugging Face, snapshot `20231101.en`). It's the easiest to get without logins or API keys, and the text is real and varied. I streamed it, so I never had to download the full multi-gigabyte dump. I grabbed 6,000 articles instead of 5,000 on purpose, to leave room for documents that get dropped during cleaning.

**Cleaning the text.** Each cleaning step is its own small function, and `clean_text` runs them in this order:

1. `strip_html` removes HTML tags using BeautifulSoup.
2. `normalize_unicode` applies NFKC normalization, so things like ligatures (`ﬁ`), non-breaking spaces and full-width letters become plain, standard characters.
3. `clean_whitespace` collapses repeated spaces and big gaps of blank lines, and trims the ends.
4. `is_english` uses langdetect to keep only English text. I seeded it so results are the same on every run, and it only reads the first 1,000 characters, which is plenty and much faster.

The language check goes last on purpose, because leftover tags or odd characters can confuse the detector. If a document ends up empty or isn't English, `clean_text` returns `None` and the pipeline drops it.

**Testing.** I wrote 12 pytest tests. Each function gets a normal case plus edge cases like empty strings, `None`, whitespace-only text, French text, and tag-only input.

### Results

| What I measured | Number |
|---|---|
| Raw documents downloaded | 6,000 |
| Raw documents containing HTML-like tags | 73 (about 1.2%) |
| Kept after cleaning | 5,982 |
| Dropped (empty or not English) | 18 |

One thing I noticed: this Wikipedia dataset is already mostly plain text, so very few documents had HTML in them. I kept `strip_html` anyway because the task requires it, it's tested, and it'll matter if I later feed in sources like ArXiv or Reddit.

The output file has three columns: `id`, `title` and `text`.

### A note about the GPU

My laptop doesn't have a CUDA-capable NVIDIA GPU, so `verify_env.py` prints `CUDA available: False` and a warning, then carries on using the CPU. I chose a warning over a hard `assert` so the script doesn't crash on machines like mine. Nothing in Week 1 needs a GPU. If a later week does, I'll use Google Colab.

### How to run it

Run everything from the project root. These are Windows PowerShell commands:

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install the dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Check the environment
python scripts/verify_env.py

# 4. Download the raw data (about 72 MB, saved to data/raw/)
python scripts/download_data.py

# 5. Clean it (writes data/processed/clean_corpus.parquet)
python scripts/run_preprocessing.py

# 6. Run the tests
python -m pytest tests -v
```

On Mac or Linux, activate the environment with `source venv/bin/activate` instead.

### Project layout

```
parallax-labs-internship/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── preprocessing.py
├── tests/
│   └── test_preprocessing.py
├── scripts/
│   ├── verify_env.py
│   ├── download_data.py
│   └── run_preprocessing.py
└── data/            (git-ignored)
    ├── raw/
    └── processed/
```

### Good to know

- `data/raw/` and `data/processed/` are git-ignored because the files are large (about 72 MB and 41 MB). Running steps 4 and 5 above recreates both.
- Run the tests from the project root with `python -m pytest tests`, otherwise Python can't find the `src` folder.
- `requirements.txt` has exact pinned versions, so installing from it gives you the same setup I used.