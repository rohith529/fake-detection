"""Unit tests for NLP preprocessing pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from preprocess import (
    fix_encoding,
    expand_contractions,
    strip_leakage,
    tokenize,
    clean,
    NEGATION_WORDS
)


def test_fix_encoding_mojibake():
    # Mojibake apostrophe â€™ should repair to '
    bad = "Doctors hate this trickâ€™s secret"
    fixed = fix_encoding(bad)
    assert "â€™" not in fixed
    assert "'" in fixed or "’" in fixed


def test_contractions_expansion():
    text = "Scientists won't confirm hot water cures cancer and can't explain it"
    expanded = expand_contractions(text)
    assert "will not" in expanded
    assert "can not" in expanded


def test_negation_words_preserved_in_cleaning():
    text = "The government did not declare a complete lockdown"
    cleaned = clean(text)
    # Ensure "not" is strictly preserved in cleaned output
    assert "not" in cleaned.split()


def test_synthetic_bias_tokens_eradicated():
    # Debunking lead and question lead headlines should NOT have __DLEAD__ or __QLEAD__
    test_cases = [
        "Fact Check: Did the prime minister announce free laptops for all students?",
        "No, UNESCO did not name the Indian anthem the best in the world",
        "False: Viral video claims massive earthquake in Delhi",
        "Why is everyone sharing this viral message about salt water?"
    ]
    for text in test_cases:
        cleaned = clean(text)
        assert "__DLEAD__" not in cleaned
        assert "__QLEAD__" not in cleaned


def test_strip_leakage():
    agency_headline = "(Reuters) - India's inflation eases to 4.5% in October"
    stripped = strip_leakage(agency_headline)
    assert "(reuters)" not in stripped.lower()
