"""Unit tests for Stance Detection and Natural Language Inference (NLI)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from similarity import classify_stance, compute_scores, compute_source_signal


def test_stance_refutes_contradiction():
    claim = "Government declares nationwide lockdown from tomorrow"
    evidence_title = "PIB Fact Check: Government has not declared nationwide lockdown, viral post is fake"
    topical_sim = 0.65

    stance, conf, agree = classify_stance(claim, evidence_title, topical_sim)
    assert stance == "REFUTES"
    assert agree < 0.20  # Debunking should result in low agreement
    assert conf >= 0.70


def test_stance_supports_entailment():
    claim = "ISRO successfully launches Aditya L1 solar observation spacecraft"
    evidence_title = "ISRO launches Aditya-L1 observatory to study the Sun"
    topical_sim = 0.75

    stance, conf, agree = classify_stance(claim, evidence_title, topical_sim)
    assert stance == "SUPPORTS"
    assert agree >= 0.70
    assert conf >= 0.70


def test_stance_neutral_unrelated():
    claim = "Stock markets hit record highs today"
    evidence_title = "Monsoon arrives early in Kerala, heavy rains forecast"
    topical_sim = 0.05

    stance, conf, agree = classify_stance(claim, evidence_title, topical_sim)
    assert stance == "NEUTRAL"
    assert agree == 0.50


def test_source_signal_aggregation_with_debunking():
    articles = [
        {"title": "Fact check: Claim is false and debunked", "source_domain": "thehindu.com", "reliability": 0.90},
        {"title": "Viral rumor is untrue, confirms official", "source_domain": "bbc.com", "reliability": 0.92}
    ]
    stance_results = [
        {"stance": "REFUTES", "confidence": 0.90, "agreement": 0.05, "topical_sim": 0.60},
        {"stance": "REFUTES", "confidence": 0.85, "agreement": 0.10, "topical_sim": 0.55}
    ]

    source_signal, conf_flag = compute_source_signal(articles, stance_results)
    assert conf_flag is True
    # Should be close to 0.0 (strongly indicating Fake)
    assert source_signal < 0.20


def test_source_signal_aggregation_with_corroboration():
    articles = [
        {"title": "Official confirmation: ISRO launches satellite", "source_domain": "thehindu.com", "reliability": 0.90},
        {"title": "ISRO announces successful launch into orbit", "source_domain": "reuters.com", "reliability": 0.95}
    ]
    stance_results = [
        {"stance": "SUPPORTS", "confidence": 0.85, "agreement": 0.85, "topical_sim": 0.70},
        {"stance": "SUPPORTS", "confidence": 0.90, "agreement": 0.90, "topical_sim": 0.75}
    ]

    source_signal, conf_flag = compute_source_signal(articles, stance_results)
    assert conf_flag is True
    # Should be high (indicating Real)
    assert source_signal > 0.80
