"""End-to-end test of the running Flask API."""
import requests, json

API = "http://127.0.0.1:5000/api/analyze"

tests = [
    # (actual_label, text, optional_url)
    ("FAKE", "SHOCKING: Scientists confirm hot water cures all cancers, doctors hate this!", ""),
    ("FAKE", "Fact Check: No, PM Modi has not announced another Covid-19 lockdown", ""),
    ("FAKE", "Viral video shows Modi performing black magic, shared widely on WhatsApp", ""),
    ("FAKE", "Fact Check: 1938 video of BKS Iyengar shared as PM Modi performing yoga", ""),
    ("FAKE", "False: Conspiracy theory claims Sushant was killed for a corona testing patent", ""),
    ("FAKE", "Misleading: Unrelated images passed off as visuals from the horrific Assam floods", ""),
    ("REAL", "Central bank keeps interest rates unchanged, citing stable inflation and steady growth", "https://www.thehindu.com/business/"),
    ("REAL", "LAC tensions: China strategy behind deliberately failing talks with India", ""),
    ("REAL", "COVID-19: India single-day spike drops to 55342 as tally approaches 72 lakh", ""),
    ("REAL", "WHO praises India Aarogya Setu app says it helped in identifying COVID-19 clusters", ""),
    ("REAL", "Hathras case: CBI reaches victim village, visits crime scene", ""),
    ("REAL", "Bihar Assembly Election 2020: Tej Pratap shifted from Mahua to Hasanpur", ""),
    ("REAL", "Mumbai BMC to book and penalise people stepping out without face masks", ""),
    ("REAL", "India has signed 250 documents on Space cooperation with 59 countries: ISRO chief", ""),
]

correct = wrong = unverified = 0
print(f"{'Status':<8} {'Act':<5} {'Pred':<12} {'Cred':>5}  {'p_real':>7}  Text")
print("-" * 100)

for actual, text, url in tests:
    payload = {"text": text, "url": url}
    r = requests.post(API, json=payload, timeout=15)
    d = r.json()
    verdict = d.get("verdict", "ERROR")
    pred = "REAL" if verdict == "Likely real" else ("FAKE" if verdict == "Likely fake" else "UNV")
    ok = pred == actual
    if ok:
        correct += 1
        mark = "OK   "
    elif pred == "UNV":
        unverified += 1
        mark = "UNV  "
    else:
        wrong += 1
        mark = "WRONG"
    print(f"[{mark}] {actual:<5} {verdict:<12} {d['credibility']:>3}/100  {d['p_real']:>7.3f}  {text[:72]}")
    if d.get("markers", {}).get("flags"):
        for f in d["markers"]["flags"]:
            print(f"         -> {f}")

print()
print(f"Results: {correct} correct / {unverified} unverified / {wrong} wrong  out of {len(tests)}")
print(f"Accuracy on definite calls: {correct}/{correct+wrong}  ({correct/(correct+wrong)*100:.1f}%)")
