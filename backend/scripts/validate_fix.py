"""
Quick validation: compare old vs new credibility formula on a balanced
sample of FAKE and REAL headlines from ifnd.csv.
"""
import sys
from pathlib import Path
import joblib, pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean
from explain import linguistic_markers

bundle = joblib.load(ROOT / "models" / "model.joblib")
vec, model = bundle["vectorizer"], bundle["model"]

df = pd.read_csv(ROOT / "data" / "ifnd.csv").dropna(subset=["text","label"])
df["label"] = df["label"].astype(int)

# Balanced sample: 500 fake + 500 real
sample = pd.concat([
    df[df["label"]==0].sample(500, random_state=42),
    df[df["label"]==1].sample(500, random_state=42),
], ignore_index=True)

sample["cleaned"] = sample["text"].map(clean)
X = vec.transform(sample["cleaned"])
proba = model.predict_proba(X)[:, 1]
sample["p_real"] = proba

def ling_real(text):
    m = linguistic_markers(text)
    return 1 - min(m["score"], 0.5)

sample["ling"] = sample["text"].map(ling_real)

# OLD formula
sample["cred_old"] = (0.65 * sample["p_real"] + 0.20 * sample["ling"]) / (0.65 + 0.20)
sample["verdict_old"] = sample["cred_old"].apply(
    lambda c: "Likely real" if c >= 0.65 else ("Likely fake" if c <= 0.40 else "Unverified"))

# NEW formula
sample["cred_new"] = (0.80 * sample["p_real"] + 0.10 * sample["ling"]) / (0.80 + 0.10)
sample["verdict_new"] = sample["cred_new"].apply(
    lambda c: "Likely real" if c >= 0.60 else ("Likely fake" if c < 0.40 else "Unverified"))

def evaluate(col, label_col="label"):
    correct = 0
    unverified = 0
    wrong = 0
    for _, row in sample.iterrows():
        v = row[col]
        if v == "Unverified":
            unverified += 1
        elif (v == "Likely real" and row[label_col] == 1) or (v == "Likely fake" and row[label_col] == 0):
            correct += 1
        else:
            wrong += 1
    n = len(sample)
    return correct, wrong, unverified, n

print("=" * 55)
print(f"{'':20s}  {'OLD':>10}  {'NEW':>10}")
print("=" * 55)

for label, name in [(None, "ALL"), (0, "FAKE only"), (1, "REAL only")]:
    sub = sample if label is None else sample[sample["label"]==label]
    n = len(sub)

    def sub_eval(col):
        c = w = u = 0
        for _, row in sub.iterrows():
            v = row[col]
            if v == "Unverified": u += 1
            elif (v=="Likely real" and row["label"]==1) or (v=="Likely fake" and row["label"]==0): c += 1
            else: w += 1
        return c, w, u

    co, wo, uo = sub_eval("verdict_old")
    cn, wn, un = sub_eval("verdict_new")

    print(f"\n{name} (n={n})")
    print(f"  Correct      : {co:4d} ({co/n*100:.1f}%)   →  {cn:4d} ({cn/n*100:.1f}%)")
    print(f"  Wrong        : {wo:4d} ({wo/n*100:.1f}%)   →  {wn:4d} ({wn/n*100:.1f}%)")
    print(f"  Unverified   : {uo:4d} ({uo/n*100:.1f}%)   →  {un:4d} ({un/n*100:.1f}%)")

print()

# Show cases the old formula got wrong but the new one gets right
old_wrong = sample[(sample["verdict_old"] != "Unverified") &
                   ~((sample["verdict_old"]=="Likely real") & (sample["label"]==1)) &
                   ~((sample["verdict_old"]=="Likely fake") & (sample["label"]==0))]
new_fixed = old_wrong[
    ((old_wrong["verdict_new"]=="Likely real") & (old_wrong["label"]==1)) |
    ((old_wrong["verdict_new"]=="Likely fake") & (old_wrong["label"]==0))
]
print(f"Cases old formula got WRONG that new formula fixes: {len(new_fixed)}")
for _, row in new_fixed.head(8).iterrows():
    actual = "REAL" if row["label"]==1 else "FAKE"
    print(f"  actual={actual}  old={row['verdict_old']:12s}  new={row['verdict_new']:12s} | {row['text'][:85]}")
