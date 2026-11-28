# Parallax Labs Internship

Hi, I'm Aleeza. This is the single repository for my whole Parallax Labs internship. Every week I'll add my new work here and update this README with what changed, what got installed, and how to run it.

---

## Week 1: Environment, Data Ingestion & Preprocessing

### What this week was about

Before you can build anything smart with text, you need three boring-but-important things: a setup that works the same on every machine, a real pile of text to work with, and a way to make that text clean and consistent. That's what I built this week.

By the end, I had a verified environment, 6,000 real Wikipedia articles, a small tested cleaning pipeline, and a clean dataset of **5,982 documents** saved as `data/processed/clean_corpus.parquet`.

### What's in the repo

- `scripts/verify_env.py` checks that the core libraries import (pandas, sentence-transformers,LangGraph, ChromaDB, spaCy), that the spaCy English model loads, and whether a CUDA GPU is available.
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
- LangGraph is installed because the Week 1 task lists it; I don’t use it yet..

---

## Week 2: Semantic Chunking, Embeddings & Vector DB

### What this week was about

Last week I ended up with clean documents. This week I made them searchable by meaning. I cut the documents into small overlapping chunks, turned each chunk into an embedding, stored the embeddings in a local ChromaDB database, and built a semantic search on top of it. Then I benchmarked the search with 10 sample questions.

I also acted on my Week 1 feedback from the mentor and added character, word and token length columns to the clean corpus.

### What's new in the repo

- `src/metrics.py` counts characters, words and tokens (tokens counted with the embedding model's own tokenizer).
- `src/chunking.py` splits documents into overlapping chunks.
- `src/embedding.py` loads the model, creates embeddings, and logs time and memory.
- `src/vectorstore.py` opens the persistent ChromaDB database and the collection (cosine distance).
- `src/search.py` is the semantic search function.
- `scripts/run_chunking.py` chunks the whole corpus and saves `data/processed/chunks.parquet`.
- `scripts/embed_sample.py` measures embedding speed and memory on 500 chunks.
- `scripts/ingest_chroma.py` embeds chunks and stores them in ChromaDB. It can be stopped and re-run.
- `scripts/search_demo.py` lets me type a question and see the top results.
- `scripts/benchmark_retrieval.py` benchmarks latency and checks the results against 10 queries.
- `results/benchmark_results.json` holds the saved benchmark output.
- `tests/test_metrics.py`, `tests/test_chunking.py` and `tests/test_search.py` add 18 new unit tests (30 in total).

### How I approached it

**Length metrics.** The mentor suggested adding length columns, so `clean_corpus.parquet` now has `char_count`, `word_count` and `token_count`. This showed that the average document has 2,644 tokens and the longest has 27,197, while the embedding model reads at most 256 tokens at a time. 4,989 of 5,982 documents (83%) are over that limit, which is why chunking is needed.

**Chunking strategy.** I used recursive character splitting. It tries to cut at paragraph breaks first, then line breaks, then the end of a sentence, then spaces, and only cuts at a fixed length as a last resort. Chunks are at most 800 characters with a 100-character overlap, so an idea sitting at a cut still appears in both neighbouring chunks. I picked this over semantic chunking because it is fast (0.7 seconds for the whole corpus), needs no model, is easy to unit-test, and I have no GPU.

**Embeddings.** I used `sentence-transformers/all-MiniLM-L6-v2` (384 numbers per chunk, 256-token limit). It is small enough to run well on a CPU. Embeddings are normalized to length 1.

**ChromaDB.** I used a persistent client that saves to `data/chroma_db/`. Every chunk is stored with its text and metadata (document id, title, position in the document). Chunks are added in batches of 500, and the ingestion script skips chunks that are already stored, so it is safe to stop and run again.

**Search.** The question is embedded with the same model, and ChromaDB returns the nearest chunks. I pass the query embeddings myself instead of letting ChromaDB embed the text, so the question always goes through the same model as the stored chunks.

### Results

| Chunking | Value |
|---|---|
| Chunks from the whole corpus | 132,595 (about 22.2 per document) |
| Average chunk length | 648 characters, about 138 tokens |
| Chunks over 256 tokens (sample of 2,000) | 4 (about 0.2%) |
| Time to chunk everything | 0.7 seconds |

| Embedding and ingestion | Value |
|---|---|
| Speed on a sample of 500 chunks | 40.4 chunks per second (24.7 ms per chunk) |
| Memory | 430 MB before the model, 903 MB after loading it, 1,034 MB after embedding |
| Chunks stored in ChromaDB | 40,825 (from the first 1,000 articles) |
| Ingestion time | 23.3 minutes at 29.2 chunks per second |
| Memory during ingestion | stayed between 1.05 and 1.25 GB |

**Retrieval benchmark** (10 questions, 5 repeats each, median time, top 5 results, 40,825 chunks):

| Measure | Value |
|---|---|
| Right article ranked first (Hit@1) | 9 of 10 |
| Right article in the top 5 (Hit@5) | 10 of 10 |
| Mean time to embed the question | 10.6 ms |
| Mean ChromaDB search time | 1.9 ms |
| Mean total per query | 12.5 ms |
| Slowest query | 14.7 ms |
| Agreement with exact brute-force search | 100% |

For the one question that did not rank first ("Which philosopher wrote Language, Truth and Logic?"), the expected article, A. J. Ayer, came second behind Aristotle. In an earlier manual search, "Who was the 16th president of the United States?" ranked Andrew Johnson above Abraham Lincoln. That matches what I read about semantic search: it is good at topic and meaning, and weaker at exact details like numbers.

### Distance metric

I used **cosine distance** (0 means the same direction, larger means less similar). ChromaDB's default is L2, so I set cosine on purpose when creating the collection. Because the embeddings are normalized, cosine and dot product rank results in the same order. The metric is chosen when the collection is created, so using a different one would mean creating a new collection.

### ChromaDB persistence and edge cases

- **Re-running ingestion:** already stored chunk ids are skipped, so nothing is added twice.
- **Adding an id that already exists:** in my test on ChromaDB 1.5.9 the second add was ignored and the stored text did not change.
- **Missing or empty database:** the helper creates the folder and an empty collection. Search returns an empty list, and the scripts print a clear message instead of crashing.
- **Collection names:** ChromaDB needs 3 to 512 characters, so the name is a single constant (`wiki_chunks`).
- **Same model everywhere:** one `MODEL_NAME` is used for chunk embeddings and for questions, because vectors from different models do not match.
- **Asking for more results than are stored:** it returns what exists, and a unit test covers it.
- **Interrupted runs:** at most the current batch of 500 is lost, and a re-run continues from the stored ids.
- **Git:** `data/chroma_db/` is git-ignored because it is large and can be rebuilt.

### Decisions and limitations

- **Only the first 1,000 articles are in the database.** Embedding all 132,595 chunks would take about 76 minutes on my CPU (my estimate from the ingestion speed). I did not run it. The same script finishes the job: `python scripts/ingest_chroma.py` without `--max-docs`.
- **The subset is not random.** The Wikipedia dataset is stored alphabetically, so these 1,000 articles all have titles starting with A. My 10 benchmark questions are about articles I knew were in the subset, so the results are probably better than they would be for random questions.
- **About 0.2% of chunks are over 256 tokens**, and the model cuts off their ends. I kept the chunk size, because smaller chunks would mean even more chunks to embed.
- **10 queries is a small test.** It is a sanity check, not a full evaluation.
- **CPU only.** There is no CUDA GPU on my machine, so everything here ran on the CPU.
- **Small leftovers in the text:** some articles have empty brackets where a pronunciation was removed, and 266 documents contain a pipe character.

Ideas for later: hybrid keyword and semantic search, a reranker, semantic chunking, comparing different chunk sizes, and embedding the full corpus.

### Dependencies added this week

`psutil` is new (it reports memory use). `chromadb`, `sentence-transformers` and `transformers` were already installed. `requirements.txt` is refreshed with pinned versions.

### How to run it

Run everything from the project root, after the Week 1 setup. The first run downloads the tokenizer and the embedding model (about 90 MB).

```powershell
# 1. Add the length columns to the clean corpus (re-runs the cleaning)
python scripts/run_preprocessing.py

# 2. Chunk the whole corpus (saves data/processed/chunks.parquet)
python scripts/run_chunking.py

# 3. Measure embedding speed and memory on 500 chunks
python scripts/embed_sample.py

# 4. Embed and store the first 1,000 articles in ChromaDB (about 23 minutes on a CPU)
python scripts/ingest_chroma.py --max-docs 1000

# 5. Try a search
python scripts/search_demo.py "What is anarchism?" --top-k 3

# 6. Run the benchmark (saves results/benchmark_results.json)
python scripts/benchmark_retrieval.py

# 7. Run the unit tests (30 tests)
python -m pytest tests -v
```

### Project layout after Week 2

```
parallax-labs-internship/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── vectorstore.py
│   └── search.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   ├── test_chunking.py
│   └── test_search.py
├── scripts/
│   ├── verify_env.py
│   ├── download_data.py
│   ├── run_preprocessing.py
│   ├── run_chunking.py
│   ├── embed_sample.py
│   ├── ingest_chroma.py
│   ├── search_demo.py
│   └── benchmark_retrieval.py
├── results/
│   └── benchmark_results.json
└── data/            (git-ignored)
    ├── raw/
    ├── processed/
    └── chroma_db/
```