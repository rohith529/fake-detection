"""Flask backend: serves the frontend and the /api/* endpoints.
Run:  python backend/app.py   ->  http://127.0.0.1:5000"""
import hashlib
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import joblib
from flask import Flask, jsonify, request, send_from_directory

import db
import learn
import retriever
import similarity
from explain import explain_words, text_characteristics
from preprocess import clean

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "model.joblib"
SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text())
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "veritas-admin-secret-2026")

# Initialize database schema and performance PRAGMAs once on startup
db.init_db()

# Model loading with thread-safe lock
_model_lock = threading.RLock()

if not MODEL_FILE.exists():
    import train
    train.main()

BUNDLE = joblib.load(MODEL_FILE)


def reload_model():
    """Thread-safe hot-reload of production model bundle."""
    global BUNDLE
    new_bundle = joblib.load(MODEL_FILE)
    with _model_lock:
        BUNDLE = new_bundle


app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")

# ── Simple In-Memory Rate Limiting ───────────────────────────────────────────
_rate_limits = {}
_rate_lock = threading.Lock()


def is_rate_limited(client_ip: str, max_requests: int = 40, window_secs: int = 60) -> bool:
    """Sliding-window IP rate limiter."""
    now = time.time()
    with _rate_lock:
        timestamps = _rate_limits.get(client_ip, [])
        valid_ts = [t for t in timestamps if now - t < window_secs]
        if len(valid_ts) >= max_requests:
            _rate_limits[client_ip] = valid_ts
            return True
        valid_ts.append(now)
        _rate_limits[client_ip] = valid_ts
        return False


def domain_of(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url if "//" in url else "//" + url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def source_score(domain: str):
    if not domain:
        return None
    for known, score in SOURCES.items():
        if domain == known or domain.endswith("." + known):
            return score
    return None


def factcheck_links(text: str):
    q = quote_plus(" ".join(text.split()[:10]))
    return [
        dict(name="Google Fact Check Explorer", link=f"https://toolbox.google.com/factcheck/explorer/search/{q}"),
        dict(name="Snopes", link=f"https://www.snopes.com/?s={q}"),
        dict(name="PIB Fact Check (India)", link="https://pib.gov.in/factcheck.aspx")
    ]


def bayesian_fusion(p_real: float, source_evidence: list[dict], url_reliability: float | None) -> float:
    """Bayesian evidence updating on log-odds space.
    
    Prior: L0 = logit(p_real)
    Evidence updates:
        SUPPORTS:  +1.2 * reliability * confidence
        REFUTES:   -2.2 * reliability * confidence
        NEUTRAL:    0.0
        URL Prior: +1.0 * (url_rel - 0.50) if URL given
    Posterior: p_final = sigmoid(L0 + sum(updates))
    """
    p_clamped = max(0.005, min(0.995, p_real))
    logit_prior = math.log(p_clamped / (1.0 - p_clamped))
    delta_L = 0.0

    for ev in source_evidence:
        stance = ev.get("stance", "NEUTRAL")
        conf = float(ev.get("confidence", 0.5))
        rel = float(ev.get("reliability", 0.8))
        if stance == "SUPPORTS":
            delta_L += 1.2 * rel * conf
        elif stance == "REFUTES":
            delta_L -= 2.2 * rel * conf

    if url_reliability is not None:
        delta_L += 1.0 * (float(url_reliability) - 0.50)

    logit_post = logit_prior + delta_L
    p_final = 1.0 / (1.0 + math.exp(-logit_post))
    return max(0.0, min(1.0, p_final))


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/analyze")
def analyze():
    client_ip = request.remote_addr or "127.0.0.1"
    if is_rate_limited(client_ip):
        return jsonify(error="Rate limit exceeded. Please wait a moment before trying again."), 429

    d = request.get_json(force=True, silent=True) or {}
    text = (d.get("text") or "").strip()
    url = (d.get("url") or "").strip()

    # Input length & DoS validation
    if len(text.split()) < 5:
        return jsonify(error="Paste a full headline or claim (at least 5 words)."), 400
    if len(text) > 4000:
        return jsonify(error="Text exceeds maximum length of 4,000 characters."), 400

    t0 = time.perf_counter()

    # 1. Thread-safe ML inference
    with _model_lock:
        vectorizer = BUNDLE["vectorizer"]
        model = BUNDLE["model"]
        model_name = BUNDLE["best"]

    cleaned_text = clean(text)
    x = vectorizer.transform([cleaned_text])
    p_real = float(model.predict_proba(x)[0][1])

    # 2. Descriptive text characteristics (no heuristic penalties)
    char_info = text_characteristics(text)

    # 3. Consolidated single-pass retrieval and stance detection
    articles = retriever.fetch(text)
    stance_scores = similarity.compute_scores(text, articles)
    source_signal, confidence_flag = similarity.compute_source_signal(articles, stance_scores)

    # Compile source evidence payload
    source_evidence = [
        {
            "title": a["title"],
            "link": a["link"],
            "source_domain": a["source_domain"],
            "reliability": a["reliability"],
            "agreement": s["agreement"],
            "stance": s["stance"],
            "confidence": s["confidence"]
        }
        for a, s in zip(articles, stance_scores)
    ]

    # 4. Domain & URL reliability check
    domain = domain_of(url)
    src_rel = source_score(domain)

    # 5. Principled Bayesian Evidence Fusion with Epistemic Corroboration
    support_count = sum(1 for e in source_evidence if e["stance"] == "SUPPORTS")
    refute_count = sum(1 for e in source_evidence if e["stance"] == "REFUTES")

    cred_prob = bayesian_fusion(p_real, source_evidence, src_rel)

    # Epistemic Corroboration Principle:
    # A static bag-of-words/n-gram classifier evaluating an extraordinary breaking claim
    # (e.g. death rumors, unverified records) cannot declare it "Likely real" if zero
    # trusted news sources corroborate it and no trusted publisher link was supplied.
    uncorroborated_flag = False
    if cred_prob >= 0.60 and support_count == 0 and (src_rel is None or src_rel < 0.80):
        cred_prob = 0.50
        uncorroborated_flag = True

    credibility_pct = int(round(cred_prob * 100))

    # Standard verdict thresholds
    if credibility_pct >= 60:
        verdict = "Likely real"
    elif credibility_pct < 40:
        verdict = "Likely fake"
    else:
        verdict = "Unverified"

    # 6. Database record update (No DDL)
    con = db.get_db()
    try:
        if domain and verdict == "Likely fake":
            con.execute(
                "INSERT INTO flags (domain, count) VALUES (?, 1) ON CONFLICT(domain) DO UPDATE SET count=count+1",
                (domain,)
            )
        row = con.execute("SELECT count FROM flags WHERE domain = ?", (domain,)).fetchone() if domain else None
        flag_count = row["count"] if row else 0

        con.execute(
            "INSERT INTO history (ts, snippet, verdict, credibility, domain) VALUES (datetime('now'), ?, ?, ?, ?)",
            (text[:120], verdict, credibility_pct, domain)
        )
        con.commit()
    finally:
        con.close()

    # Reuse already-fetched articles for related links
    related = [
        {"title": a["title"], "link": a["link"], "source": a["source_domain"]}
        for a in articles[:5]
    ] if verdict != "Likely real" else []

    explanation = explain_words(text, vectorizer, model)

    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    return jsonify(
        verdict=verdict,
        credibility=credibility_pct,
        p_real=round(p_real, 3),
        model=model_name,
        latency_ms=latency_ms,
        markers=char_info,
        explanation=explanation,
        source=dict(
            domain=domain,
            score=src_rel,
            flagged=flag_count,
            watchlist=bool(flag_count >= 3)
        ),
        related=related,
        factchecks=factcheck_links(text),
        source_evidence=source_evidence,
        support_count=support_count,
        refute_count=refute_count,
        uncorroborated_flag=uncorroborated_flag,
        confidence_flag=confidence_flag,
        source_signal=round(source_signal, 3)
    )


@app.get("/api/history")
def history():
    con = db.get_db()
    try:
        rows = [dict(r) for r in con.execute("SELECT * FROM history ORDER BY id DESC LIMIT 8")]
        return jsonify(rows)
    finally:
        con.close()


# ── Protected Live Learning & Feedback ───────────────────────────────────────

def _retrain_job():
    if learn.retrain():
        reload_model()


@app.post("/api/feedback")
def feedback():
    """Accepts user feedback and stages it for verification without poisoning."""
    d = request.get_json(force=True, silent=True) or {}
    text = d.get("text") or ""
    label = d.get("label")
    client_ip = request.remote_addr or ""
    ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:12]

    ok = learn.add_feedback(text, label, ip_hash=ip_hash)
    return jsonify(saved=ok, **learn.status())


@app.post("/api/retrain")
def retrain():
    """Trigger background model retraining (admin authenticated)."""
    auth_key = request.headers.get("X-Admin-Key") or (request.get_json(force=True, silent=True) or {}).get("admin_key")
    if auth_key != ADMIN_API_KEY:
        return jsonify(error="Unauthorized: Administrative authentication required."), 401

    if learn.state["running"]:
        return jsonify(started=False, message="Retraining is already in progress.", **learn.status()), 409

    harvested = 0
    if (request.get_json(force=True, silent=True) or {}).get("harvest", False):
        try:
            harvested = learn.harvest()
        except Exception as e:
            _log.warning("Harvest failed: %s", e)

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
