import pandas as pd

from src.metrics import add_length_metrics, count_tokens, get_tokenizer


def test_add_length_metrics_columns_and_values():
    df = pd.DataFrame({"text": ["Hello world, this is a test.", "Short"]})
    out = add_length_metrics(df)
    assert {"char_count", "word_count", "token_count"} <= set(out.columns)
    assert out["char_count"].tolist() == [28, 5]
    assert out["word_count"].tolist() == [6, 1]
    assert out["token_count"].tolist() == [8, 1]


def test_add_length_metrics_does_not_change_input():
    df = pd.DataFrame({"text": ["abc"]})
    add_length_metrics(df)
    assert "char_count" not in df.columns


def test_count_tokens_same_for_any_batch_size():
    tokenizer = get_tokenizer()
    texts = ["one two three"] * 5
    assert count_tokens(texts, tokenizer, batch_size=2) == count_tokens(
        texts, tokenizer, batch_size=10
    )