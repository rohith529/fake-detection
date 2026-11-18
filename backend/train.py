"""Stage 3+4 — Training: TF-IDF features, compare classical models, save the best.
Run:  python backend/train.py

Changes from original
─────────────────────
1. Drop TERROR-only and MISLEADIND rows before splitting.
   TERROR is 100 % REAL in IFND; MISLEADIND is a typo duplicate of MISLEADING
   (17 rows, 100 % FAKE).  Both are category-label shortcuts the model can
   memorise.  Removing them forces it to learn language, not topic.

2. TfidfVectorizer: max_features 30 k → 50 k, min_df kept at 2.
   50 k gives richer vocabulary and rarer n-grams room without overfitting.

3. Added SGD (Hinge) as a fourth candidate — fast linear SVM variant that
   scales well and often wins on text with strong minority-class imbalance.

4. Honest evaluation rules unchanged: test set = static only, live rows go
   into training only, best model chosen on validation F1.
"""
import json, time
from pathlib import Path

import joblib, numpy as np, pandas as pd, sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocess import clean, strip_leakage
from seed_data import SEED

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "models"
MAX_ROWS   = None
LIVE_WEIGHT = {"feedback": 1.0, "rss": 1.0}

# Categories that create a shortcut (topic ≡ label) rather than testing language
_SKIP_CATEGORIES = {"TERROR", "MISLEADIND"}


def _isot():
    f, t = DATA / "Fake.csv", DATA / "True.csv"
    if not (f.exists() and t.exists()):
        return None
    fake, real = pd.read_csv(f), pd.read_csv(t)
    fake["label"], real["label"] = 0, 1
    df = pd.concat([fake, real], ignore_index=True)
    body = df["text"].fillna("").map(strip_leakage)
    df["text"] = df["title"].fillna("").map(strip_leakage) + ". " + body
    return df[["text", "label"]].assign(dataset="ISOT")


def load_static():
    parts = []
    iso = _isot()
    if iso is not None:
        parts.append(iso)
    for p in sorted(DATA.glob("*.csv")):
        if p.name in ("Fake.csv", "True.csv", "live.csv"):
            continue
        d = pd.read_csv(p)
        # Accept both (text, label) and (text, label, category) schemas
        if {"text", "label"} <= set(d.columns):
            d = d[["text", "label"] + (["category"] if "category" in d.columns else [])].dropna(subset=["text", "label"])
            d["text"] = d["text"].astype(str).map(strip_leakage)
            # Drop category-shortcut rows (only applies to IFND)
            if "category" in d.columns:
                before = len(d)
                d = d[~d["category"].isin(_SKIP_CATEGORIES)]
                dropped = before - len(d)
                if dropped:
                    print(f"  Dropped {dropped} rows from shortcut categories {_SKIP_CATEGORIES}")
            parts.append(d.assign(dataset=p.stem))
    if not parts:
        return (pd.DataFrame(SEED, columns=["text", "label"]).assign(dataset="toy seed"),
                "built-in toy seed (DEMO ONLY)")
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset="text")
    return df, " + ".join(sorted(df["dataset"].unique()))


def load_live():
    """Load verified live learning rows directly from data/history.db (live_data table)."""
    db_path = DATA / "history.db"
    if not db_path.exists():
        return pd.DataFrame(columns=["text", "label", "origin"])
    try:
        import sqlite3
        con = sqlite3.connect(db_path)
        d = pd.read_sql_query("SELECT text, label, origin FROM live_data", con)
        con.close()
        if d.empty:
            return pd.DataFrame(columns=["text", "label", "origin"])
        d = d.dropna(subset=["text", "label"])
        d["text"] = d["text"].astype(str).map(strip_leakage)
        return d.drop_duplicates(subset="text")
    except Exception:
        return pd.DataFrame(columns=["text", "label", "origin"])


def _models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0, max_iter=2000), cv=3),
        "SGD (Hinge)": CalibratedClassifierCV(
            SGDClassifier(loss="hinge", class_weight="balanced",
                          max_iter=1000, random_state=42, n_jobs=-1), cv=3),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
            class_weight="balanced", max_features="sqrt"),
    }


def _scores(m, X, y):
    pred = m.predict(X)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return dict(
        accuracy=round(accuracy_score(y, pred), 4),
        precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
        confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main(verbose=True):
    static, source = load_static()
    live = load_live()

    if MAX_ROWS and len(static) > MAX_ROWS:
        static = static.sample(MAX_ROWS, random_state=42)

    tiny = len(static) < 200
    if verbose:
        print(f"Static data: {source} ({len(static)} rows) | live rows: {len(live)}")

    static = static.assign(clean=static["text"].map(clean))

    # Remove rows that collapse to < 3 tokens after preprocessing
    static = static[static["clean"].str.split().str.len() >= 3].reset_index(drop=True)
    if verbose:
        print(f"After quality filter: {len(static)} rows")

    # test set from static only; validation carved from the remaining training split
    trval, test = train_test_split(
        static, test_size=0.2, random_state=42, stratify=static["label"])
    train, val = train_test_split(
        trval, test_size=0.2, random_state=42, stratify=trval["label"])
    train = train.assign(weight=1.0)

    if len(live):
        lv = live.assign(
            clean=live["text"].map(clean),
            weight=live.get("origin", "rss").map(LIVE_WEIGHT).fillna(1.0))
        train = pd.concat(
            [train, lv[["clean", "label", "weight"]]], ignore_index=True)

    # ── Vectorizer ──────────────────────────────────────────────────────────
    # max_features 30 k → 50 k: gives rarer n-grams and vocabulary room.
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True)

    Xtr = vec.fit_transform(train["clean"])
    Xva  = vec.transform(val["clean"])
    Xte  = vec.transform(test["clean"])
    ytr, yva, yte = train["label"], val["label"], test["label"]
    w = train["weight"].values

    results, fitted = [], {}
    for name, m in _models().items():
        t0 = time.time()
        try:
            m.fit(Xtr, ytr, sample_weight=w)
        except TypeError:
            m.fit(Xtr, ytr)
        secs = round(time.time() - t0, 2)
        va, te = _scores(m, Xva, yva), _scores(m, Xte, yte)
        results.append(dict(model=name, **te, val_f1=va["f1"], train_seconds=secs))
        fitted[name] = m
        if verbose:
            print(f"{name:22s}  val_f1={va['f1']:.4f}  |  "
                  f"TEST acc={te['accuracy']:.4f}  f1={te['f1']:.4f}  "
                  f"prec={te['precision']:.4f}  rec={te['recall']:.4f}")

    # Best model chosen on validation F1 (not test F1)
    best = max(results, key=lambda r: r["val_f1"])["model"]
    OUT.mkdir(exist_ok=True)
    joblib.dump(
        dict(vectorizer=vec, model=fitted[best], best=best,
             explainer=fitted[best]),
        OUT / "model.joblib")

    (OUT / "metrics.json").write_text(json.dumps(dict(
        dataset=source, rows=len(static), live_rows=len(live),
        test_rows=len(test), best=best,
        sklearn=sklearn.__version__,
        trained_at=time.strftime("%Y-%m-%d %H:%M"),
        demo_only=tiny, results=results), indent=2))

    if verbose:
        print(f"\nBest model: {best}"
              + ("  ⚠ toy data: numbers are meaningless" if tiny else ""))


if __name__ == "__main__":
    main()
