"""Similarity & Stance Detection Engine: Classifies evidence into Entailment, Contradiction, or Neutral.

Remediated architecture:
1. Stance Detection:
   Classifies retrieved articles into:
     - SUPPORTS (Entailment): Directly corroborates the claim's core assertion.
     - REFUTES (Contradiction): Explicitly denies, debunks, or contradicts the claim.
     - NEUTRAL: Unrelated or adjacent topic without affirming or refuting.
2. Subject-Entity Sensitivity:
   Prevents false corroboration when the action applies to a different entity
   (e.g., "KCR's sister passes away" must not corroborate "KCR passed away").
3. Honest Confidence Flagging:
   confidence_flag is only True if there is definitive corroboration or refutation,
   never merely because uninformative neutral articles were found.
"""
import logging
import os
import re
from typing import Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean, expand_contractions

_MODE = os.environ.get("SIMILARITY_MODE", "lightweight").lower()
_log = logging.getLogger(__name__)

# Refutation and debunking markers
REFUTATION_MARKERS = frozenset({
    "debunk", "debunks", "debunked", "debunking",
    "false", "hoax", "deny", "denies", "denied", "denial",
    "refute", "refutes", "refuted", "fake", "rumor", "rumour",
    "rumors", "rumours", "myth", "myths", "baseless", "untrue",
    "no evidence", "misleading", "fact check", "factcheck",
    "fact-check", "not true", "busts", "busted", "clarifies",
    "clarification", "fabricated", "unfounded", "scam", "falsely"
})

SUPPORT_MARKERS = frozenset({
    "confirms", "confirmed", "announces", "announced", "official",
    "verified", "approves", "approved", "signs", "launches",
    "launched", "declares", "declared", "passes", "passed"
})

_TRANSFORMER_MODEL = None
if _MODE == "semantic":
    try:
        from sentence_transformers import SentenceTransformer
        _TRANSFORMER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _log.info("Loaded sentence-transformers model for semantic similarity.")
    except Exception as e:
        _log.warning("sentence-transformers unavailable; falling back to hybrid stance. %s", e)
        _TRANSFORMER_MODEL = None


def _check_entity_mismatch(claim: str, evidence_title: str) -> bool:
    """Detect if evidence is about a relative or different entity rather than the claim's subject."""
    claim_low = claim.lower()
    title_low = evidence_title.lower()

    # Relative / modifier patterns in title not present in claim
    relatives = ["'s sister", "'s brother", "'s son", "'s daughter", "'s mother", "'s father", "'s wife", "'s aide", "ex-mla", "writer"]
    for rel in relatives:
        if rel in title_low and rel not in claim_low:
            return True
    return False


def classify_stance(claim: str, evidence_title: str, topical_similarity: float) -> Tuple[str, float, float]:
    """Classify the relationship between claim and evidence headline.
    
    Returns:
        (stance, confidence, agreement_score)
        stance: "SUPPORTS" | "REFUTES" | "NEUTRAL"
        confidence: [0.0, 1.0]
        agreement_score: [0.0, 1.0] where 1.0 is strong support, 0.0 is strong refutation, 0.5 is neutral.
    """
    claim_norm = expand_contractions(claim.lower())
    title_norm = expand_contractions(evidence_title.lower())
    title_words = set(re.findall(r"[a-z]+", title_norm))

    # 1. Subject entity divergence check
    if _check_entity_mismatch(claim_norm, title_norm):
        return "NEUTRAL", 0.85, 0.50

    # 2. Check for refutation / debunking markers in evidence
    refutation_hits = [m for m in REFUTATION_MARKERS if m in title_norm or m in title_words]
    support_hits = [m for m in SUPPORT_MARKERS if m in title_words]

    # Topical relevance threshold
    if topical_similarity < 0.20:
        return "NEUTRAL", 0.70, 0.50

    # Contradiction / Refutation
    if refutation_hits:
        refute_conf = min(1.0, 0.50 + 0.50 * topical_similarity + 0.10 * len(refutation_hits))
        agreement = max(0.0, 0.15 * (1.0 - topical_similarity))  # Drops to 0.0
        return "REFUTES", round(refute_conf, 2), round(agreement, 2)

    # Entailment / Corroboration: requires substantial topical similarity
    if topical_similarity >= 0.45 or (topical_similarity >= 0.32 and support_hits):
        support_conf = min(1.0, 0.40 + 0.60 * topical_similarity)
        agreement = min(1.0, 0.50 + 0.50 * topical_similarity)
        return "SUPPORTS", round(support_conf, 2), round(agreement, 2)

    # Moderate overlap without explicit corroboration
    return "NEUTRAL", 0.50, 0.50


def compute_scores(claim_text: str, articles: list[dict]) -> list[dict]:
    """Compute stance and agreement for each retrieved article."""
    if not articles:
        return []

    titles = [a.get("title", "") for a in articles]

    if _TRANSFORMER_MODEL is not None:
        try:
            claim_emb = _TRANSFORMER_MODEL.encode([claim_text])[0]
            title_embs = _TRANSFORMER_MODEL.encode(titles)
            sims = cosine_similarity([claim_emb], title_embs)[0]
            topical_sims = [float(max(0.0, min(1.0, s))) for s in sims]
        except Exception:
            topical_sims = None
    else:
        topical_sims = None

    if topical_sims is None:
        cleaned_claim = clean(claim_text)
        cleaned_titles = [clean(t) for t in titles]
        corpus = [cleaned_claim] + cleaned_titles
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        tfidf_mat = vec.fit_transform(corpus)
        sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:])[0]
        topical_sims = [float(max(0.0, min(1.0, s))) for s in sims]

    results = []
    for art, top_sim in zip(articles, topical_sims):
        stance, conf, agree = classify_stance(claim_text, art.get("title", ""), top_sim)
        results.append({
            "stance": stance,
            "confidence": conf,
            "agreement": agree,
            "topical_sim": round(top_sim, 2)
        })

    return results


def compute_source_signal(articles: list[dict], stance_results: list[dict]) -> Tuple[float, bool]:
    """Aggregate stance-classified evidence into a unified Source Signal [0.0, 1.0].
    
    confidence_flag is True ONLY when at least one article has a clear stance
    (SUPPORTS or REFUTES) with meaningful confidence.
    """
    if not articles or not stance_results:
        return 0.5, False

    weighted_signals = []
    total_weights = 0.0
    has_informative_evidence = False

    for art, res in zip(articles, stance_results):
        rel = art.get("reliability", 0.80)
        stance = res["stance"]
        conf = res["confidence"]
        top_sim = res["topical_sim"]

        if top_sim < 0.15:
            continue

        weight = rel * conf
        if stance == "REFUTES":
            signal = 0.0
            weight *= 2.0
            has_informative_evidence = True
        elif stance == "SUPPORTS":
            signal = 1.0
            weight *= 1.5
            has_informative_evidence = True
        else:
            signal = 0.5
            weight *= 0.2

        weighted_signals.append(signal * weight)
        total_weights += weight

    if total_weights == 0.0:
        return 0.5, False

    source_signal = sum(weighted_signals) / total_weights
    confidence_flag = has_informative_evidence

    return float(max(0.0, min(1.0, source_signal))), confidence_flag
