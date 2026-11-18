"""Unit tests for active production model explanation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import BUNDLE
from explain import explain_words, get_production_feature_weights


def test_weights_derived_from_production_model():
    model = BUNDLE["model"]
    vectorizer = BUNDLE["vectorizer"]
    n_features = len(vectorizer.get_feature_names_out())

    weights = get_production_feature_weights(model, n_features)
    assert len(weights) == n_features
    # Check that weights are non-trivial
    assert (weights != 0).sum() > 100


def test_explain_words_output_structure():
    text = "The Reserve Bank of India kept interest rates unchanged amid inflation"
    vectorizer = BUNDLE["vectorizer"]
    model = BUNDLE["model"]

    res = explain_words(text, vectorizer, model, top=5)
    assert "tokens" in res
    assert "fake_terms" in res
    assert "real_terms" in res
    assert len(res["tokens"]) > 0
    # Every token dict must contain 't' and 's'
    for tk in res["tokens"]:
        assert "t" in tk
        assert "s" in tk
