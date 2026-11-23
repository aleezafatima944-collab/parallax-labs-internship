from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_tokenizer(model_name=MODEL_NAME):
    """Load the tokenizer used by the embedding model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 10**9  # we only count tokens, so silence length warnings
    return tokenizer


def count_tokens(texts, tokenizer, batch_size=256):
    """Return the number of tokens in each text."""
    counts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        counts.extend(len(ids) for ids in encoded)
    return counts


def add_length_metrics(df, tokenizer=None):
    """Return a copy of the DataFrame with char_count, word_count and token_count."""
    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    tokenizer = tokenizer or get_tokenizer()
    df["token_count"] = count_tokens(df["text"].tolist(), tokenizer)
    return df