"""Flask backend: serves the frontend and the /api/* endpoints.
Run:  python backend/app.py   ->  http://127.0.0.1:5000"""
import json, sqlite3, time
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import feedparser, joblib, requests
from flask import Flask, jsonify, request, send_from_directory

import threading

import retriever
import similarity
from explain import explain_words, linguistic_markers
from preprocess import clean

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE, DB_FILE = ROOT / "models" / "model.joblib", ROOT / "data" / "history.db"
SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text())

if not MODEL_FILE.exists():                      # first run: train automatically
    import train; train.main()
BUNDLE = joblib.load(MODEL_FILE)

def reload_model():
    global BUNDLE
    BUNDLE = joblib.load(MODEL_FILE)
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")

# ---------- SQLite: query history + unreliable-source watchlist ----------
def db():
    con = sqlite3.connect(DB_FILE); con.row_factory = sqlite3.Row
    con.executescript("""CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, ts TEXT,
        snippet TEXT, verdict TEXT, credibility INTEGER, domain TEXT);
        CREATE TABLE IF NOT EXISTS flags(domain TEXT PRIMARY KEY, count INTEGER);""")
    return con

def domain_of(url: str) -> str:
    if not url: return ""
    host = urlparse(url if "//" in url else "//" + url).netloc.lower()
    return host[4:] if host.startswith("www.") else host

def source_score(domain: str):
    for known, score in SOURCES.items():
        if domain == known or domain.endswith("." + known): return score
    return None

# ---------- Authentic-news recommendation ----------
def related_articles(text: str, n: int = 5):
    return [{"title": a["title"], "link": a["link"], "source": a["source_domain"]}
            for a in retriever.fetch(text)[:n]]

def factcheck_links(text: str):
    q = quote_plus(" ".join(text.split()[:12]))
    return [dict(name="Google Fact Check Explorer", link=f"https://toolbox.google.com/factcheck/explorer/search/{q}"),
            dict(name="Snopes", link=f"https://www.snopes.com/?s={q}"),
            dict(name="PIB Fact Check (India)", link="https://pib.gov.in/factcheck.aspx")]

# ---------- API ----------
@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/analyze")
def analyze():
    d = request.get_json(force=True, silent=True) or {}
    text, url = (d.get("text") or "").strip(), (d.get("url") or "").strip()
    if len(text.split()) < 5:
        return jsonify(error="Paste a full headline or article (at least 5 words)."), 400

    t0 = time.perf_counter()
    x = BUNDLE["vectorizer"].transform([clean(text)])
    p_real = float(BUNDLE["model"].predict_proba(x)[0][1])
    markers = linguistic_markers(text)
    articles = retriever.fetch(text)
    scores   = similarity.compute_scores(text, articles)
    source_signal, confidence_flag = similarity.compute_source_signal(articles, scores)
    domain = domain_of(url)
    src = source_score(domain)

    # ── Fusion Engine ─────────────────────────────────────────────────────────
    # Step 1: ling+ML blend (unchanged from previous version)
    ling_score  = markers["score"]              # 0 = clean, 1 = strong fake signal
    ling_weight = min(0.60, ling_score * 0.8)  # scales 0→0, 0.75→0.60, 1.0→0.60
    ml_weight   = 1.0 - ling_weight
    ling_real   = 1.0 - ling_score             # invert: high flag → low real prob

    # Blended ML+ling value (used as ML signal into the fusion step)
    p_real_adj = ml_weight * p_real + ling_weight * ling_real

    # Step 2: fuse with Source_Signal when we have enough live evidence
    if confidence_flag and ling_score <= 0.70:
        # Fused formula: ML 55 %, Source 45 %
        cred = 0.55 * p_real_adj + 0.45 * source_signal
    else:
        # Existing formula: ling+ML blend, optional URL source nudge
        parts = [(ml_weight, p_real), (ling_weight, ling_real)]
        if src is not None:
            src_w = 0.05
            parts = [(w * (1 - src_w), v) for w, v in parts]
            parts.append((src_w, src))
        cred = sum(w * v for w, v in parts) / sum(w for w, _ in parts)

    cred = max(0.0, min(1.0, cred))  # clamp to [0, 1]

    # Verdict thresholds match the CSS gradient bands on the gauge
    verdict = "Likely real" if cred >= 0.60 else "Likely fake" if cred < 0.40 else "Unverified"

    con = db()
    if domain and verdict == "Likely fake":
        con.execute("INSERT INTO flags VALUES(?,1) ON CONFLICT(domain) DO UPDATE SET count=count+1", (domain,))
    row = con.execute("SELECT count FROM flags WHERE domain=?", (domain,)).fetchone() if domain else None
    con.execute("INSERT INTO history(ts,snippet,verdict,credibility,domain) VALUES(datetime('now'),?,?,?,?)",
                (text[:120], verdict, round(cred * 100), domain))
    con.commit(); con.close()

    return jsonify(
        verdict=verdict, credibility=round(cred * 100), p_real=round(p_real, 3),
        model=BUNDLE["best"], latency_ms=round((time.perf_counter() - t0) * 1000),
        markers=markers, explanation=explain_words(text, BUNDLE["vectorizer"], BUNDLE["explainer"]),
        source=dict(domain=domain, score=src, flagged=row["count"] if row else 0,
                    watchlist=bool(row and row["count"] >= 3)),
        related=related_articles(text) if verdict != "Likely real" else [],
        factchecks=factcheck_links(text),
        source_evidence=[
            {**{k: a[k] for k in ("title", "link", "source_domain", "reliability")},
             "agreement": round(s, 2)}
            for a, s in zip(articles, scores)
        ],
        confidence_flag=confidence_flag,
        source_signal=round(source_signal, 3))

@app.get("/api/history")
def history():
    con = db()
    rows = [dict(r) for r in con.execute("SELECT * FROM history ORDER BY id DESC LIMIT 8")]
    con.close(); return jsonify(rows)

# ---------- Live learning ----------
def _retrain_job():
    if learn.retrain(): reload_model()

@app.post("/api/feedback")
def feedback():
    d = request.get_json(force=True, silent=True) or {}
    ok = learn.add_feedback(d.get("text") or "", d.get("label"))
    return jsonify(saved=ok, **learn.status())

@app.post("/api/retrain")
def retrain():
    if learn.state["running"]: return jsonify(started=False, **learn.status()), 409
    harvested = 0
    if (request.get_json(force=True, silent=True) or {}).get("harvest", True):
        try: harvested = learn.harvest()
        except Exception: pass                      # offline: retrain on what we already have
    learn.state["running"] = True
    threading.Thread(target=_retrain_job, daemon=True).start()
    return jsonify(started=True, harvested=harvested, **learn.status())

@app.get("/api/learn/status")
def learn_status():
    return jsonify(learn.status())

@app.get("/api/stats")
def stats():
    return send_from_directory(ROOT / "models", "metrics.json")

if __name__ == "__main__":
    app.run(debug=False, port=5000)
