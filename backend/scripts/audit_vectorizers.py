"""Audit: cross-validate different vectorizer configs with LR and SVM."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import pandas as pd, numpy as np
from pathlib import Path
from preprocess import clean
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "ifnd.csv").dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
df["cleaned"] = df["text"].map(clean)
df = df[df["cleaned"].str.split().str.len() >= 3].reset_index(drop=True)
X, y = df["cleaned"], df["label"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    "BASE (1,2) 30k min2": TfidfVectorizer(ngram_range=(1,2), max_features=30000, min_df=2, sublinear_tf=True),
    "NEW  (1,3) 50k min2": TfidfVectorizer(ngram_range=(1,3), max_features=50000, min_df=2, sublinear_tf=True),
    "NEW  (1,2) 50k min1": TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=1, sublinear_tf=True),
    "NEW  (1,2) 60k min2": TfidfVectorizer(ngram_range=(1,2), max_features=60000, min_df=2, sublinear_tf=True),
}

clfs = {
    "LR" : LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
    "SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=1.0, max_iter=2000), cv=3),
    "LR C=5": LogisticRegression(max_iter=1000, class_weight="balanced", C=5.0),
    "SVM C=5": CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=5.0, max_iter=2000), cv=3),
}

print(f"{'Config':<24}  {'Classifier':<10}  {'F1 mean':>8}  {'F1 std':>8}")
print("-" * 60)
for vec_name, vec in configs.items():
    for clf_name, clf in clfs.items():
        pipe = Pipeline([("vec", vec), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)
        print(f"{vec_name:<24}  {clf_name:<10}  {scores.mean():.4f}    {scores.std():.4f}")
    print()
