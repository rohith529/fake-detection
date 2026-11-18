"""Offline evaluation: compare ML-only vs Source-only vs Fused credibility scoring.

Remediated architecture:
- Employs Bayesian log-odds evidence fusion instead of arbitrary hardcoded linear weights.
- Uses Natural Language Inference (NLI) stance detection (Corroboration vs. Refutation).
- Tracks fallbacks, coverage rates, and confusion matrices across ML-only, Source-only, and Fused systems.

Usage:
    python backend/evaluate_grounded.py data/ifnd.csv --sample 500
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import retriever
import similarity
from app import bayesian_fusion
from preprocess import clean

BUNDLE = joblib.load(ROOT / "models" / "model.joblib")
vec = BUNDLE["vectorizer"]
model = BUNDLE["model"]


def _ml_signal(text: str) -> float:
    return float(model.predict_proba(vec.transform([clean(text)]))[0][1])


def _verdict(prob: float) -> int:
    score = int(round(prob * 100))
    if score >= 60:
        return 1
    if score < 40:
        return 0
    return 0  # Unverified -> conservative: treat as fake


def _metrics(labels, preds, name: str) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return dict(
        config=name,
        accuracy=round(accuracy_score(labels, preds), 4),
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f1, 4),
        confusion=confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    )


def evaluate(csv_path: Path, sample: int = None) -> None:
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    rows = df.to_dict("records")
    if sample and sample < len(rows):
        random.seed(42)
        rows = random.sample(rows, sample)

    print(f"Evaluating {len(rows)} rows from {csv_path.name} ...")
    t_start = time.time()

    ml_preds, src_preds, fused_preds, true_labels = [], [], [], []
    fallback_count = 0

    for i, row in enumerate(rows, 1):
        text = str(row["text"])
        label = int(row["label"])
        true_labels.append(label)

        p_real = _ml_signal(text)

        try:
            arts = retriever.fetch(text)
            stance_scores = similarity.compute_scores(text, arts)
            src_sig, conf_flag = similarity.compute_source_signal(arts, stance_scores)
            
            evidence_payload = [
                {
                    "stance": s["stance"],
                    "reliability": a.get("reliability", 0.85),
                    "confidence": s["confidence"]
                }
                for a, s in zip(arts, stance_scores)
            ]
            if not arts:
                raise ValueError("No coverage found")
        except Exception:
            fallback_count += 1
            src_sig, conf_flag = 0.5, False
            evidence_payload = []

        # ML-only verdict (unfused)
        ml_preds.append(_verdict(p_real))

        # Source-only verdict (threshold at 0.50)
        src_preds.append(1 if src_sig >= 0.5 else 0)

        # Principled Bayesian Fused verdict
        fused_prob = bayesian_fusion(p_real, evidence_payload, url_reliability=None)
        fused_preds.append(_verdict(fused_prob))

        if i % 25 == 0 or i == len(rows):
            elapsed = time.time() - t_start
            print(f"  {i}/{len(rows)} rows | {elapsed:.1f}s elapsed | Fallback count: {fallback_count}")

    results = [
        _metrics(true_labels, ml_preds, "ml_only"),
        _metrics(true_labels, src_preds, "source_only"),
        _metrics(true_labels, fused_preds, "bayesian_fused"),
    ]

    out = dict(
        dataset=csv_path.stem,
        rows=len(df),
        sample=sample if sample else len(df),
        fallback_count=fallback_count,
        evaluated_at=time.strftime("%Y-%m-%d %H:%M"),
        results=results
    )

    out_path = ROOT / "models" / "grounded_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print(f"{'Config':<18}  {'Accuracy':>9}  {'Precision':>10}  {'Recall':>7}  {'F1':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:<18}  {r['accuracy']:>9.4f}  {r['precision']:>10.4f}  "
              f"{r['recall']:>7.4f}  {r['f1']:>7.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="Path to evaluation dataset CSV")
    parser.add_argument("--sample", type=int, default=None, help="Sample size to evaluate")
    args = parser.parse_args()
    evaluate(args.csv_path, args.sample)
