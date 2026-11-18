"""Before/after report: old model (original preprocess) vs new model (fixed preprocess).
Runs entirely from saved artefacts — does not re-train."""
import sys, warnings, importlib, types, copy
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd, numpy as np, joblib
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

# ── Reload the CURRENT (new) preprocess module ────────────────────────────────
import preprocess as new_pre
importlib.reload(new_pre)

# ── Reconstruct OLD clean() without the new features ─────────────────────────
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_stemmer_old = PorterStemmer()
_URL_old = re.compile(r"https?://\S+|www\.\S+")

def clean_old(text: str) -> str:
    tokens = re.findall(r"[a-z]+", _URL_old.sub(" ", text.lower()))
    return " ".join(_stemmer_old.stem(t) for t in tokens
                    if t not in ENGLISH_STOP_WORDS and len(t) > 2)

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "ifnd.csv").dropna(subset=["text","label"])
df["label"] = df["label"].astype(int)
# Evaluate on the FULL set (including shortcut categories) so the comparison
# reflects what a user submitting any headline would experience
print(f"Full dataset: {len(df):,} rows")
print(f"  Real={( df['label']==1).sum():,}   Fake={(df['label']==0).sum():,}")
print()

# ── New model predictions ─────────────────────────────────────────────────────
new_bundle = joblib.load(ROOT / "models" / "model.joblib")
df["clean_new"] = df["text"].map(new_pre.clean)
X_new  = new_bundle["vectorizer"].transform(df["clean_new"])
pred_n = new_bundle["model"].predict(X_new)
prob_n = new_bundle["model"].predict_proba(X_new)[:, 1]

def metrics(y, pred):
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    pf, rf, f1f, _ = precision_recall_fscore_support(y, pred, labels=[0], average=None, zero_division=0)
    return dict(acc=acc, prec=p, rec=r, f1=f1, tn=tn, fp=fp, fn=fn, tp=tp,
                fake_prec=pf[0], fake_rec=rf[0], fake_f1=f1f[0])

m_new = metrics(df["label"], pred_n)

print("=" * 65)
print(f"{'Metric':<30}  {'NEW':>10}")
print("=" * 65)
print(f"  {'Accuracy':<28}  {m_new['acc']*100:>9.2f}%")
print(f"  {'F1 (REAL)':<28}  {m_new['f1']:>10.4f}")
print(f"  {'Precision (REAL)':<28}  {m_new['prec']:>10.4f}")
print(f"  {'Recall (REAL)':<28}  {m_new['rec']:>10.4f}")
print(f"  {'F1 (FAKE)':<28}  {m_new['fake_f1']:>10.4f}")
print(f"  {'Precision (FAKE)':<28}  {m_new['fake_prec']:>10.4f}")
print(f"  {'Recall (FAKE)':<28}  {m_new['fake_rec']:>10.4f}")
print(f"  {'False negatives (FAKE→REAL)':<28}  {m_new['fn']:>10,}")
print(f"  {'False positives (REAL→FAKE)':<28}  {m_new['fp']:>10,}")
print()

print("Full classification report (new model):")
print(classification_report(df["label"], pred_n,
      target_names=["FAKE","REAL"], digits=4, zero_division=0))

print()
print("Per-category accuracy (new model):")
df["pred_n"] = pred_n
cats = df.groupby("category").apply(
    lambda g: pd.Series({
        "n": len(g), "fake": (g["label"]==0).sum(), "real": (g["label"]==1).sum(),
        "correct": (g["label"]==g["pred_n"]).sum(),
        "acc": round((g["label"]==g["pred_n"]).mean(), 4),
    }), include_groups=False).sort_values("acc")
print(cats.to_string())
print()

# ── Confidence distribution ───────────────────────────────────────────────────
df["p_real_n"] = prob_n
print("Confidence distribution (new model):")
for lbl, name in [(0,"FAKE"),(1,"REAL")]:
    sub = df[df["label"]==lbl]["p_real_n"]
    print(f"  {name}  avg={sub.mean():.3f}  median={sub.median():.3f}  "
          f"<50%: {(sub<0.5).sum():,}  >50%: {(sub>0.5).sum():,}")
