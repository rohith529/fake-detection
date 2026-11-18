"""Error analysis: what does the current model get wrong and why?"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import pandas as pd, numpy as np, joblib
from pathlib import Path
from preprocess import clean
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "ifnd.csv").dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
df["cleaned"] = df["text"].map(clean)
df = df[df["cleaned"].str.split().str.len() >= 3].reset_index(drop=True)

bundle = joblib.load(ROOT / "models" / "model.joblib")
vec, model = bundle["vectorizer"], bundle["model"]
X = vec.transform(df["cleaned"])
df["pred"]   = model.predict(X)
df["p_real"] = model.predict_proba(X)[:, 1]
df["correct"] = df["pred"] == df["label"]

# ----- per-category breakdown -----
print("=== PER-CATEGORY ACCURACY ===")
cat = df.groupby(["category", "label"]).apply(
    lambda g: pd.Series({"n": len(g), "acc": (g["correct"]).mean()}), include_groups=False
).reset_index()
print(cat.to_string(index=False))
print()

# ----- false negatives: FAKE predicted as REAL -----
fn = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("p_real", ascending=False)
print(f"=== FALSE NEGATIVES (FAKE → REAL), n={len(fn)} ===")
print(f"Top confidence wrong calls:")
for _, r in fn.head(20).iterrows():
    print(f"  [{r.p_real:.3f}] cat={r.category:<12} | {r.text[:90]}")
print()

# ----- false positives: REAL predicted as FAKE -----
fp = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("p_real")
print(f"=== FALSE POSITIVES (REAL → FAKE), n={len(fp)} ===")
print("Lowest confidence real calls:")
for _, r in fp.head(20).iterrows():
    print(f"  [{r.p_real:.3f}] cat={r.category:<12} | {r.text[:90]}")
print()

# ----- what tokens drive wrong calls -----
# For false negatives: what are the most common words appearing in them?
fn_words = " ".join(fn["cleaned"].tolist())
real_words = " ".join(df[df["label"] == 0]["cleaned"].tolist())
from collections import Counter
fn_counts  = Counter(fn_words.split())
all_counts = Counter(real_words.split())
# rate: how often this word appears in FN vs all fake
print("=== TOP TOKENS IN FALSE NEGATIVES (not in top fake vocab) ===")
fn_tokens = sorted(fn_counts.items(), key=lambda x: -x[1])
for tok, cnt in fn_tokens[:30]:
    all_c = all_counts.get(tok, 0)
    rate = cnt / max(all_c, 1)
    print(f"  {tok:<20} fn={cnt:4d}  all_fake={all_c:5d}  fn_rate={rate:.3f}")

# ----- word count analysis of errors -----
print()
print("=== WORD COUNT DISTRIBUTION OF ERRORS ===")
df["wc"] = df["text"].str.split().str.len()
print("False negatives word count:")
print(fn["wc"].describe().round(1))
print("False positives word count:")
print(fp["wc"].describe().round(1))
print("All fake word count:")
print(df[df["label"]==0]["wc"].describe().round(1))
