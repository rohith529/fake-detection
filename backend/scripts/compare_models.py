"""Compare old saved model vs freshly trained model on the full IFND dataset.
Excludes the shortcut categories (TERROR, MISLEADIND) for a fair comparison.
"""
import sys, warnings, shutil
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import pandas as pd, numpy as np, joblib
from pathlib import Path
from preprocess import clean
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent

# ── Load dataset ─────────────────────────────────────────────────────────────
df_full = pd.read_csv(ROOT / "data" / "ifnd.csv").dropna(subset=["text", "label"])
df_full["label"] = df_full["label"].astype(int)

# Fair evaluation: exclude shortcut categories
df = df_full[~df_full["category"].isin({"TERROR", "MISLEADIND"})].copy().reset_index(drop=True)
print(f"Evaluation set: {len(df):,} rows  (excluded TERROR+MISLEADIND shortcut categories)")
print(f"  Real: {(df['label']==1).sum():,}   Fake: {(df['label']==0).sum():,}")
print()

# ── New model ─────────────────────────────────────────────────────────────────
new_bundle = joblib.load(ROOT / "models" / "model.joblib")
df["cleaned_new"] = df["text"].map(clean)         # uses updated preprocess.py
X_new = new_bundle["vectorizer"].transform(df["cleaned_new"])
pred_new   = new_bundle["model"].predict(X_new)
proba_new  = new_bundle["model"].predict_proba(X_new)[:, 1]

def report(label, preds, probas):
    acc = accuracy_score(label, preds)
    p, r, f1, _ = precision_recall_fscore_support(label, preds, average="binary", zero_division=0)
    cm = confusion_matrix(label, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    # At 50% threshold
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision  : {p:.4f}")
    print(f"  Recall     : {r:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print()
    print(classification_report(label, preds, target_names=["FAKE","REAL"],
                                 digits=4, zero_division=0))
    return acc, p, r, f1, fn, fp

print("=" * 60)
print("NEW MODEL (fixed pipeline)")
print("=" * 60)
nacc, np_, nr, nf1, nfn, nfp = report(df["label"], pred_new, proba_new)

# ── Per-category ──────────────────────────────────────────────────────────────
df["pred_new"] = pred_new
print("=" * 60)
print("PER-CATEGORY  (new model)")
print("=" * 60)
cats = df.groupby("category").apply(
    lambda g: pd.Series({
        "n"       : len(g),
        "fake"    : (g["label"]==0).sum(),
        "real"    : (g["label"]==1).sum(),
        "correct" : (g["label"] == g["pred_new"]).sum(),
        "acc"     : round((g["label"] == g["pred_new"]).mean(), 4),
    }), include_groups=False
).sort_values("acc")
print(cats.to_string())
print()

# ── Fake / Real split ─────────────────────────────────────────────────────────
print("=" * 60)
print("FAKE vs REAL (new model)")
print("=" * 60)
for lbl, name in [(0, "FAKE"), (1, "REAL")]:
    sub = df[df["label"] == lbl]
    correct = (sub["pred_new"] == lbl).sum()
    pct = correct / len(sub) * 100
    print(f"  {name}:  {correct:,} / {len(sub):,} correct  ({pct:.2f}%)")

print()
print("=" * 60)
print("HARDEST FALSE NEGATIVES (missed FAKE — high p_real)")
print("=" * 60)
df["p_real_new"] = proba_new
fn_df = df[(df["label"]==0) & (df["pred_new"]==1)].sort_values("p_real_new", ascending=False)
print(f"Total false negatives: {len(fn_df)}")
for _, r in fn_df.head(15).iterrows():
    print(f"  [{r.p_real_new:.3f}] {r.category:<12} | {r.text[:85]}")

print()
print("=" * 60)
print("HARDEST FALSE POSITIVES (missed REAL — low p_real)")
print("=" * 60)
fp_df = df[(df["label"]==1) & (df["pred_new"]==0)].sort_values("p_real_new")
print(f"Total false positives: {len(fp_df)}")
for _, r in fp_df.head(15).iterrows():
    print(f"  [{r.p_real_new:.3f}] {r.category:<12} | {r.text[:85]}")
