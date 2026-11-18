"""
Full evaluation of the saved model against the ifnd.csv dataset.
Covers both FAKE (label=0) and REAL (label=1) news across all categories.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))   # so preprocess imports cleanly

from preprocess import clean

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = ROOT / "models" / "model.joblib"
bundle     = joblib.load(MODEL_PATH)
vec        = bundle["vectorizer"]
model      = bundle["model"]
best_name  = bundle["best"]

print(f"Loaded model : {best_name}")
print(f"Model path   : {MODEL_PATH}\n")

# ── Load dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "ifnd.csv").dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

print(f"Dataset rows : {len(df):,}")
print(f"  Real (1)   : {(df['label']==1).sum():,}")
print(f"  Fake (0)   : {(df['label']==0).sum():,}")
print()

# ── Preprocess & predict ──────────────────────────────────────────────────────
print("Preprocessing & predicting … (may take a moment)")
df["cleaned"] = df["text"].map(clean)
X = vec.transform(df["cleaned"])
preds   = model.predict(X)
probas  = model.predict_proba(X)[:, 1]   # P(real)

df["predicted"] = preds
df["confidence"] = np.where(preds == 1, probas, 1 - probas)

# ── Overall metrics ───────────────────────────────────────────────────────────
print("=" * 60)
print("OVERALL METRICS")
print("=" * 60)
acc = accuracy_score(df["label"], preds)
p, r, f1, _ = precision_recall_fscore_support(
    df["label"], preds, average="binary", zero_division=0)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {p:.4f}")
print(f"  Recall    : {r:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print()
print(classification_report(df["label"], preds,
                             target_names=["FAKE", "REAL"], zero_division=0))

cm = confusion_matrix(df["label"], preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix (rows=actual, cols=predicted):")
print(f"  {'':12s}  Pred FAKE  Pred REAL")
print(f"  Actual FAKE   {tn:8,}   {fp:8,}")
print(f"  Actual REAL   {fn:8,}   {tp:8,}")
print()

# ── Per-category breakdown ────────────────────────────────────────────────────
if "category" in df.columns:
    print("=" * 60)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 60)
    cats = df.groupby("category").apply(
        lambda g: pd.Series({
            "total": len(g),
            "real": (g["label"] == 1).sum(),
            "fake": (g["label"] == 0).sum(),
            "correct": (g["label"] == g["predicted"]).sum(),
            "accuracy": round((g["label"] == g["predicted"]).mean(), 4),
        }), include_groups=False
    ).sort_values("accuracy")
    print(cats.to_string())
    print()

# ── Fake news deep-dive ───────────────────────────────────────────────────────
fake_df = df[df["label"] == 0].copy()
print("=" * 60)
print(f"FAKE NEWS RESULTS  (n={len(fake_df):,})")
print("=" * 60)
fake_acc = (fake_df["label"] == fake_df["predicted"]).mean()
missed   = fake_df[fake_df["predicted"] == 1]   # predicted REAL but actually FAKE
correct  = fake_df[fake_df["predicted"] == 0]
print(f"  Correctly identified as FAKE : {len(correct):,}  ({fake_acc*100:.2f}%)")
print(f"  Missed (predicted REAL)      : {len(missed):,}  ({(1-fake_acc)*100:.2f}%)")
print()
print("── Sample CORRECTLY detected FAKE headlines ──")
for _, row in correct.head(10).iterrows():
    print(f"  [{row['confidence']*100:.0f}% conf] {row['text'][:100]}")
if len(missed):
    print()
    print("── Missed FAKE headlines (model said REAL) ──")
    for _, row in missed.head(10).iterrows():
        print(f"  [{row['confidence']*100:.0f}% conf] {row['text'][:100]}")
print()

# ── Real news deep-dive ───────────────────────────────────────────────────────
real_df = df[df["label"] == 1].copy()
print("=" * 60)
print(f"REAL NEWS RESULTS  (n={len(real_df):,})")
print("=" * 60)
real_acc  = (real_df["label"] == real_df["predicted"]).mean()
false_pos = real_df[real_df["predicted"] == 0]   # predicted FAKE but actually REAL
correct_r = real_df[real_df["predicted"] == 1]
print(f"  Correctly identified as REAL : {len(correct_r):,}  ({real_acc*100:.2f}%)")
print(f"  False positives (pred FAKE)  : {len(false_pos):,}  ({(1-real_acc)*100:.2f}%)")
print()
print("── Sample CORRECTLY detected REAL headlines ──")
for _, row in correct_r.head(10).iterrows():
    print(f"  [{row['confidence']*100:.0f}% conf] {row['text'][:100]}")
if len(false_pos):
    print()
    print("── False positives: REAL flagged as FAKE ──")
    for _, row in false_pos.head(10).iterrows():
        print(f"  [{row['confidence']*100:.0f}% conf] {row['text'][:100]}")
print()

# ── Confidence distribution ───────────────────────────────────────────────────
print("=" * 60)
print("CONFIDENCE DISTRIBUTION")
print("=" * 60)
for label, name in [(0, "FAKE"), (1, "REAL")]:
    sub = df[df["label"] == label]["confidence"]
    print(f"  {name}  avg={sub.mean():.3f}  min={sub.min():.3f}  "
          f"max={sub.max():.3f}  <50%: {(sub < 0.5).sum()}")

# ── Low-confidence predictions ────────────────────────────────────────────────
low = df[df["confidence"] < 0.60].sort_values("confidence")
if len(low):
    print()
    print(f"Low-confidence predictions (< 60%, n={len(low)}):")
    for _, row in low.head(10).iterrows():
        actual = "REAL" if row["label"] == 1 else "FAKE"
        pred   = "REAL" if row["predicted"] == 1 else "FAKE"
        mark   = "✓" if row["label"] == row["predicted"] else "✗"
        print(f"  {mark} [{row['confidence']*100:.0f}%] actual={actual} pred={pred} | {row['text'][:90]}")

print()
print("Done.")
