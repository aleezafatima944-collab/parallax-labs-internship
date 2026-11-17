from src.preprocessing import (
    strip_html,
    normalize_unicode,
    clean_whitespace,
    is_english,
    clean_text,
)


def test_strip_html_removes_tags():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_empty_input():
    assert strip_html("") == ""
    assert strip_html(None) == ""


def test_normalize_unicode_ligature_and_spaces():
    assert normalize_unicode("ﬁne\u00a0Ａ") == "fine A"


def test_normalize_unicode_empty_input():
    assert normalize_unicode("") == ""

def test_clean_whitespace_collapses_spaces_and_blank_lines():
    assert clean_whitespace("  Hello     world  \n\n\n\n  Next line  ") == "Hello world\n\nNext line"


def test_clean_whitespace_empty_input():
    assert clean_whitespace("") == ""
    assert clean_whitespace("   \n  ") == ""


def test_is_english_true_for_english():
    assert is_english("This is a simple English sentence about history.") is True


def test_is_english_false_for_french():
    assert is_english("Ceci est une phrase en français.") is False


def test_is_english_false_for_empty():
    assert is_english("") is False
    assert is_english("   ") is False


def test_clean_text_full_pipeline():
    raw = "<p>Hello   <b>world</b>, this is a   simple English sentence.</p>"
    assert clean_text(raw) == "Hello world, this is a simple English sentence."


def test_clean_text_drops_non_english():
    assert clean_text("Ceci est une phrase en français.") is None


def test_clean_text_drops_empty():
    assert clean_text("") is None
    assert clean_text("<p>   </p>") is None