"""Unit tests for Bayesian Log-Odds Evidence Fusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import bayesian_fusion


def test_bayesian_fusion_no_evidence_preserves_ml():
    # When no sources are retrieved and no URL is provided, posterior = ML prior
    for p_ml in [0.20, 0.50, 0.85]:
        p_fused = bayesian_fusion(p_real=p_ml, source_evidence=[], url_reliability=None)
        assert abs(p_fused - p_ml) < 0.01


def test_bayesian_fusion_refutation_overrides_high_ml():
    # If ML model incorrectly gave 0.70 (Likely Real), but trusted sources refute it:
    evidence = [
        {"stance": "REFUTES", "reliability": 0.90, "confidence": 0.90},
        {"stance": "REFUTES", "reliability": 0.95, "confidence": 0.85}
    ]
    p_fused = bayesian_fusion(p_real=0.70, source_evidence=evidence, url_reliability=None)
    # The debunking evidence should pull the posterior sharply down into Fake territory (< 0.40)
    assert p_fused < 0.35


def test_bayesian_fusion_corroboration_overrides_low_ml():
    # If ML model was uncertain or pessimistic (0.45), but trusted sources confirm it:
    evidence = [
        {"stance": "SUPPORTS", "reliability": 0.90, "confidence": 0.85},
        {"stance": "SUPPORTS", "reliability": 0.95, "confidence": 0.90}
    ]
    p_fused = bayesian_fusion(p_real=0.45, source_evidence=evidence, url_reliability=None)
    # Supporting evidence should pull the posterior up into Real territory (>= 0.60)
    assert p_fused >= 0.70


def test_bayesian_fusion_satire_url_prior():
    # An unverified claim paired with an Infowars/The Onion link (reliability 0.05)
    p_fused = bayesian_fusion(p_real=0.50, source_evidence=[], url_reliability=0.05)
    assert p_fused < 0.45
