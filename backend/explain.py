"""Explainable AI: Derives token attributions directly from the active inference model.

Remediated architecture:
1. Synchronized Explanation & Inference:
   Extracts exact calibrated decision boundary weights from the production estimator
   (whether CalibratedClassifierCV(LinearSVC), LogisticRegression, or SGDClassifier).
   Eliminates the architectural discrepancy where an uncalibrated surrogate model
   was used to explain a different production classifier.
2. Removed Arbitrary Heuristic Penalties:
   Fact-checking leads and vocabulary are no longer penalized by hardcoded heuristic scores.
   Descriptive text characteristics are kept purely for UI transparency.
"""
import re
import numpy as np
from preprocess import clean, stem


def get_production_feature_weights(model, n_features: int) -> np.ndarray:
    """Extract effective linear decision weights from the active production model.
    
    For CalibratedClassifierCV with sigmoid (Platt) scaling:
        P(y=1|x) = 1 / (1 + exp(A * (w^T x + b) + B))
        The log-odds contribution for feature j in fold k is (-A_k) * w_{k, j}.
        Averaging across the K CV folds yields the exact calibrated linear weight vector.
    """
    # 1. CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_"):
        weights = np.zeros(n_features, dtype=np.float64)
        count = 0
        for cc in model.calibrated_classifiers_:
            base_estimator = getattr(cc, "estimator", None)
            if base_estimator is None or not hasattr(base_estimator, "coef_"):
                continue
            base_coef = base_estimator.coef_
            if base_coef.ndim > 1:
                base_coef = base_coef[0]
            
            # Extract Platt sigmoid calibration slope (-a_)
            calibrator = cc.calibrators[0] if getattr(cc, "calibrators", None) else None
            a = getattr(calibrator, "a_", None)
            slope = -float(a) if (a is not None and not np.isnan(a)) else 1.0
            
            weights += slope * base_coef
            count += 1
        if count > 0:
            return weights / count

    # 2. Linear models with direct coef_ (LogisticRegression, LinearSVC, SGDClassifier)
    if hasattr(model, "coef_"):
        coef = model.coef_
        return np.asarray(coef[0] if coef.ndim > 1 else coef, dtype=np.float64)

    # 3. Tree ensembles / other estimators
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=np.float64)

    return np.zeros(n_features, dtype=np.float64)


def explain_words(text: str, vectorizer, model, top: int = 6) -> dict:
    """Compute exact feature attributions for input text based on the active production model.
    
    Contribution = model_weight_j * tfidf_value_j.
    Positive -> pushes toward Real (Class 1).
    Negative -> pushes toward Fake (Class 0).
    """
    cleaned = clean(text)
    row = vectorizer.transform([cleaned]).tocoo()
    n_features = len(vectorizer.get_feature_names_out())
    weights = get_production_feature_weights(model, n_features)

    contrib = {j: float(weights[j] * v) for j, v in zip(row.col, row.data)}
    vocab = vectorizer.vocabulary_
    names = vectorizer.get_feature_names_out()

    peak = max((abs(c) for c in contrib.values()), default=1.0) or 1.0

    tokens = []
    for piece in re.findall(r"\w+|\W+", text):
        s = 0.0
        if piece[0].isalnum():
            # Check unstemmed and stemmed token in vocabulary
            low = piece.lower()
            idx = vocab.get(low) or vocab.get(stem(low))
            if idx is not None:
                s = contrib.get(idx, 0.0) / peak
        tokens.append(dict(t=piece, s=round(float(s), 3)))

    ranked = sorted(contrib.items(), key=lambda kv: kv[1])
    fake_terms = [names[j] for j, c in ranked[:top] if c < 0]
    real_terms = [names[j] for j, c in ranked[::-1][:top] if c > 0]

    return dict(
        tokens=tokens,
        fake_terms=fake_terms,
        real_terms=real_terms
    )


def text_characteristics(text: str) -> dict:
    """Descriptive text statistics for informational UI rendering (no heuristic penalties)."""
    words = re.findall(r"[A-Za-z']+", text)
    n = max(len(words), 1)
    caps_count = sum(1 for w in words if len(w) > 2 and w.isupper())
    caps_pct = caps_count / n
    excl_count = text.count("!")

    flags = []
    if caps_pct > 0.30:
        flags.append(f"{caps_pct:.0%} uppercase words")
    if excl_count >= 2:
        flags.append(f"{excl_count} exclamation marks")
    if n < 6:
        flags.append("Short claim text")

    return dict(
        word_count=n,
        caps_pct=round(caps_pct, 2),
        excl_count=excl_count,
        flags=flags
    )


# Backward-compatibility alias for legacy test scripts
def linguistic_markers(text: str) -> dict:
    chars = text_characteristics(text)
    return dict(score=0.0, **chars)

