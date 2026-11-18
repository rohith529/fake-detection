import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import joblib
from preprocess import clean
from explain import linguistic_markers

b = joblib.load(ROOT / "models" / "model.joblib")
vec, model = b["vectorizer"], b["model"]

tests = [
    ("FAKE", "SHOCKING: Scientists confirm hot water cures all cancers, doctors hate this!"),
    ("FAKE", "Fact Check: No, PM Modi has not announced another Covid-19 lockdown"),
    ("FAKE", "Who are these impostors in Shaheen Bagh?"),
    ("FAKE", "No, Donald Trump has not asked people to vote for BJP"),
    ("FAKE", "Viral video shows Modi performing black magic, shared widely on WhatsApp"),
    ("FAKE", "Fact Check: 1938 video of BKS Iyengar shared as PM Modi performing yoga"),
    ("FAKE", "False: Conspiracy theory claims Sushant was killed for a corona testing patent"),
    ("FAKE", "Misleading: Unrelated images passed off as visuals from the horrific Assam floods"),
    ("REAL", "Central bank keeps interest rates unchanged, citing stable inflation and steady growth"),
    ("REAL", "LAC tensions: China strategy behind deliberately failing talks with India"),
    ("REAL", "COVID-19: India single-day spike drops to 55342 as tally approaches 72 lakh"),
    ("REAL", "WHO praises India Aarogya Setu app says it helped in identifying COVID-19 clusters"),
    ("REAL", "Hathras case: CBI reaches victim village, visits crime scene"),
    ("REAL", "Bihar Assembly Election 2020: This is why Tej Pratap shifted from Mahua to Hasanpur"),
    ("REAL", "Mumbai: BMC to book, penalise people stepping out without face masks"),
]

correct = 0
print(f"{'Status':<8} {'Act':<5} {'Pred':<5} {'Cred':>5}  {'p_real':>7}  {'Ling':>5}  Text")
print("-" * 100)
for actual, text in tests:
    p_real = model.predict_proba(vec.transform([clean(text)]))[0][1]
    m = linguistic_markers(text)
    ling_score = m["score"]
    ling_weight = min(0.60, ling_score * 0.8)
    ml_weight = 1.0 - ling_weight
    ling_real = 1.0 - ling_score
    cred = (ml_weight * p_real + ling_weight * ling_real) / (ml_weight + ling_weight)
    if cred >= 0.60:
        verdict = "Likely real"
    elif cred < 0.40:
        verdict = "Likely fake"
    else:
        verdict = "Unverified"
    pred = "REAL" if verdict == "Likely real" else ("FAKE" if verdict == "Likely fake" else "UNV")
    ok = pred == actual
    if ok:
        correct += 1
    mark = "OK   " if ok else "WRONG"
    print(f"[{mark}] {actual:<5} {pred:<5} {cred*100:>4.0f}/100  {p_real:>7.3f}  {ling_score:>5}  {text[:72]}")
    for f in m["flags"]:
        print(f"         -> {f}")

print()
print(f"Score: {correct}/{len(tests)}")
