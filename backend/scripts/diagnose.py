import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import joblib
from preprocess import clean
from explain import linguistic_markers

ROOT = Path(__file__).resolve().parent.parent
b = joblib.load(ROOT / "models" / "model.joblib")
vec, model = b["vectorizer"], b["model"]

tests = [
    ("Fake", "SHOCKING: Scientists confirm hot water cures all cancers, doctors hate this!"),
    ("Fake", "Fact Check: No, PM Modi has not announced another Covid-19 lockdown"),
    ("Fake", "Who are these impostors in Shaheen Bagh?"),
    ("Fake", "No, Donald Trump hasn't asked people to vote for BJP"),
    ("Fake", "Viral video shows Modi performing black magic, shared widely on WhatsApp"),
    ("Real", "Central bank keeps interest rates unchanged, citing stable inflation"),
    ("Real", "LAC tensions: China strategy behind deliberately failing talks with India"),
    ("Real", "COVID-19: India single-day spike drops to 55342 as tally approaches 72 lakh"),
    ("Real", "WHO praises India Aarogya Setu app says it helped in identifying COVID-19 clusters"),
    ("Real", "Hathras case: CBI reaches victim village, visits crime scene"),
]

print(f"{'Status':<8} {'Actual':<6} {'Pred':<6} {'P_real':>7}  {'Ling':>5}  Text")
print("-" * 90)
for actual, text in tests:
    p = model.predict_proba(vec.transform([clean(text)]))[0][1]
    pred = "REAL" if p >= 0.5 else "FAKE"
    ok = pred == actual.upper()
    m = linguistic_markers(text)
    mark = "OK " if ok else "WRONG"
    print(f"[{mark}] {actual:<6} {pred:<6} {p:>7.3f}  {m['score']:>5}  {text[:72]}")
    if m["flags"]:
        for f in m["flags"]:
            print(f"         flags: {f}")
