"""Flask backend: serves the frontend and the /api/* endpoints.
Run:  python backend/app.py   ->  http://127.0.0.1:5000"""
from datetime import datetime, timezone
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
from explain import explain_words, text_characteristics
import gate
import learn
from preprocess import clean
import retriever
import similarity
import verify_llm

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "model.joblib"
SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text())
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
ENABLE_LEARNING = os.environ.get("ENABLE_LEARNING", "false").lower() in ("true", "1", "yes")

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


def is_rate_limited(client_ip: str, max_requests: int = 60, window_secs: int = 60) -> bool:
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


def determine_source_reputation(domain: str) -> str:
    if not domain:
        return "unknown"
    if any(domain == s or domain.endswith("." + s) for s in verify_llm.SATIRE_DOMAINS):
        return "satire"
    if any(domain == d or domain.endswith("." + d) for d in verify_llm.DISREPUTABLE_DOMAINS):
        return "unreliable"
    rel = source_score(domain)
    if rel is not None:
        return "trusted" if rel >= 0.70 else "unreliable"
    return "unknown"


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
        SUPPORTS:  +1.8 * reliability * confidence
        REFUTES:   -2.5 * reliability * confidence
        NEUTRAL:    0.0
        URL Prior: If reputable (>= 0.70): +2.5 * (url_rel - 0.50)
                   If satire / disreputable (<= 0.10): -3.5
                   Otherwise: +1.5 * (url_rel - 0.50)
    Posterior: p_final = sigmoid(L0 + sum(updates))
    """
    p_clamped = max(0.005, min(0.995, p_real))
    logit_prior = math.log(p_clamped / (1.0 - p_clamped))
    delta_L = 0.0

    for ev in source_evidence:
        stance = str(ev.get("stance", "NEUTRAL")).upper()
        conf = float(ev.get("confidence", 0.5))
        rel = float(ev.get("reliability", 0.8))
        if stance in ("SUPPORTS", "SUPPORTED"):
            delta_L += 1.8 * rel * conf
        elif stance in ("REFUTES", "REFUTED"):
            delta_L -= 2.5 * rel * conf

    if url_reliability is not None:
        rel_f = float(url_reliability)
        if rel_f <= 0.10:
            delta_L -= 3.5  # Decisive satire / fake penalty
        elif rel_f >= 0.70:
            delta_L += 2.5 * (rel_f - 0.50)  # Strong trusted news corroboration boost
        else:
            delta_L += 1.5 * (rel_f - 0.50)

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
    if not text:
        return jsonify(error="Please provide a headline or claim to analyze."), 400
    if len(text.split()) < 5:
        return jsonify(error="Paste a full headline or claim (at least 5 words)."), 400
    if len(text) > 4000:
        return jsonify(error="Text exceeds maximum length of 4,000 characters."), 400

    t0 = time.perf_counter()

    domain = domain_of(url) if url else ""
    src_rel = source_score(domain)
    source_reputation = determine_source_reputation(domain)

    # 1. Thread-safe ML inference & token explainability
    with _model_lock:
        vectorizer = BUNDLE["vectorizer"]
        model = BUNDLE["model"]
        model_name = BUNDLE["best"]

    cleaned_text = clean(text)
    x = vectorizer.transform([cleaned_text])
    p_real = float(model.predict_proba(x)[0][1])

    char_info = text_characteristics(text)
    explanation = explain_words(text, vectorizer, model)

    style_label = "neutral" if p_real > 0.50 else "sensational"
    style_indicator = {
        "label": style_label,
        "score": round(p_real, 3),
        "note": "Writing style only, not a measure of truth."
    }

    # 2. Gate classification (checks questions, predictions, opinions, too_short)
    gate_res = gate.classify_input(text)
    input_type = gate_res["type"]
    category_info = gate.classify_category(text)

    # Non-claims early return
    if input_type != "claim":
        latency_ms = int(round((time.perf_counter() - t0) * 1000))
        con = db.get_db()
        try:
            con.execute(
                "INSERT INTO history (ts, snippet, verdict, credibility, domain, confidence, claim, reasoning) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)",
                (text[:120], "NOT_A_CLAIM", 0, domain or None, "high", text[:120], gate_res["reason"][:200])
            )
            con.commit()
        finally:
            con.close()

        return jsonify(
            verdict="NOT_A_CLAIM",
            verdict_badge="NOT_A_CLAIM",
            credibility=0,
            p_real=round(p_real, 3),
            model=model_name,
            latency_ms=latency_ms,
            markers=char_info,
            explanation=explanation,
            source=dict(domain=domain, score=src_rel, flagged=0, watchlist=False),
            related=[],
            factchecks=factcheck_links(text),
            source_evidence=[],
            support_count=0,
            refute_count=0,
            uncorroborated_flag=False,
            confidence_flag=False,
            source_signal=0.0,
            claim=text,
            input_type=input_type,
            category=category_info,
            confidence="high",
            reasoning=gate_res["reason"],
            evidence=[],
            guardrail_notes=[f"Input classified as {input_type}; bypassed external verification."],
            style_indicator=style_indicator,
            source_domain=domain or None,
            source_reputation=source_reputation,
            cached=False,
            checked_at=datetime.now(timezone.utc).isoformat()
        )

    # 3. Grounded Verification & Evidence Retrieval
    v_res = verify_llm.verify_claim(text, url=url)
    v_claim = v_res.get("claim", text) if v_res else text

    # If Gemini returned a verified result
    if v_res and v_res.get("verdict") not in ("UNAVAILABLE", None):
        v_verdict = v_res.get("verdict")
        v_conf = v_res.get("confidence", "low")
        v_reasoning = v_res.get("reasoning", "")
        v_evidence = v_res.get("evidence", [])
        v_guardrails = v_res.get("guardrail_notes", [])
        v_cached = v_res.get("cached", False)
        v_checked_at = v_res.get("checked_at", datetime.now(timezone.utc).isoformat())

        if v_verdict == "SUPPORTED":
            credibility_pct = 95 if v_conf == "high" else 88
            verdict_display = "Likely real"
        elif v_verdict == "REFUTED":
            credibility_pct = 10 if v_conf == "high" else 20
            verdict_display = "Likely fake"
        elif v_verdict == "NOT_A_CLAIM":
            credibility_pct = 0
            verdict_display = "NOT_A_CLAIM"
        else:
            credibility_pct = 50
            verdict_display = "Unverified"

        source_evidence = [
            {
                "title": e.get("title", ""),
                "link": e.get("url", ""),
                "source_domain": verify_llm.extract_domain(e.get("url", "")) or e.get("publisher", ""),
                "reliability": 0.95 if str(e.get("stance", "")).lower() == "supports" else 0.85,
                "agreement": 0.90 if str(e.get("stance", "")).lower() == "supports" else (0.10 if str(e.get("stance", "")).lower() == "refutes" else 0.50),
                "stance": "SUPPORTS" if str(e.get("stance", "")).lower() == "supports" else ("REFUTES" if str(e.get("stance", "")).lower() == "refutes" else "NEUTRAL"),
                "confidence": 0.90,
                "published_date": e.get("published_date", "unknown"),
                "key_sentence": e.get("key_sentence", "")
            }
            for e in v_evidence
        ]
        support_count = sum(1 for e in source_evidence if e["stance"] == "SUPPORTS")
        refute_count = sum(1 for e in source_evidence if e["stance"] == "REFUTES")
        confidence_flag = True if v_conf in ("high", "medium") else False
        uncorroborated_flag = False
        source_signal = 0.95 if v_verdict == "SUPPORTED" else (0.10 if v_verdict == "REFUTED" else 0.50)

    else:
        # Fallback to Google News RSS + Enhanced Bayesian Evidence Fusion
        articles = retriever.fetch(text)
        stance_scores = similarity.compute_scores(text, articles)
        source_signal, confidence_flag = similarity.compute_source_signal(articles, stance_scores)

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

        support_count = sum(1 for e in source_evidence if e["stance"] == "SUPPORTS")
        refute_count = sum(1 for e in source_evidence if e["stance"] == "REFUTES")

        cred_prob = bayesian_fusion(p_real, source_evidence, src_rel)

        # General uncorroborated daily activity safeguard:
        # Factual claims in weather, sports, entertainment, politics, or general news
        # with ZERO supporting publisher evidence and NO verified source URL cannot be confirmed as real.
        uncorroborated_flag = False
        if support_count == 0 and (src_rel is None or src_rel < 0.70):
            if cred_prob < 0.40:
                # Strong sensational clickbait / hoax linguistic signals
                credibility_pct = int(round(cred_prob * 100))
                verdict_display = "Likely fake"
                v_verdict = "REFUTED"
                v_conf = "medium"
                v_reasoning = (
                    f"Sensational / Uncorroborated {category_info['label']}: Exaggerated linguistic signals and zero "
                    f"corroboration from trusted news publishers indicate fabricated clickbait or misinformation."
                )
            else:
                # Plausible/formal tone but zero external corroboration -> cap at 50% Unverified
                uncorroborated_flag = True
                cred_prob = min(0.50, cred_prob)
                credibility_pct = int(round(cred_prob * 100))
                verdict_display = "Unverified"
                v_verdict = "NOT_ENOUGH_EVIDENCE"
                v_conf = "low"
                v_reasoning = (
                    f"Uncorroborated {category_info['label']}: No corroborating reports from trusted news publishers were found "
                    f"for this event. Without verified evidence, daily activity claims cannot be confirmed as real regardless of grammatical style."
                )
        else:
            credibility_pct = int(round(cred_prob * 100))
            if credibility_pct >= 60:
                verdict_display = "Likely real"
                v_verdict = "SUPPORTED"
                v_conf = "high" if (support_count > 0 or (src_rel and src_rel >= 0.70)) else "medium"
                v_reasoning = (
                    f"Corroborated {category_info['label']}: Verified by external news reports and trusted publisher attribution."
                )
            elif credibility_pct < 40:
                verdict_display = "Likely fake"
                v_verdict = "REFUTED"
                v_conf = "high" if refute_count > 0 else "medium"
                v_reasoning = (
                    f"Refuted {category_info['label']}: Debunking markers or substantial contradiction detected."
                )
            else:
                verdict_display = "Unverified"
                v_verdict = "NOT_ENOUGH_EVIDENCE"
                v_conf = "low"
                v_reasoning = (
                    f"Inconclusive {category_info['label']}: Available publisher evidence does not definitively confirm or refute this claim."
                )

        v_evidence = [
            {
                "url": e["link"],
                "publisher": e["source_domain"],
                "title": e["title"],
                "published_date": e.get("published_date", "unknown"),
                "stance": e["stance"].lower(),
                "key_sentence": e["title"]
            }
            for e in source_evidence
        ]
        v_guardrails = [
            f"Evaluated with daily category guardrails ({category_info['label']})."
        ] if uncorroborated_flag else []
        v_cached = False
        v_checked_at = datetime.now(timezone.utc).isoformat()

    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    # 4. Database record update
    con = db.get_db()
    try:
        flag_count = 0
        if domain:
            if verdict_display == "Likely fake" or v_verdict == "REFUTED":
                con.execute(
                    "INSERT INTO flags (domain, count) VALUES (?, 1) ON CONFLICT(domain) DO UPDATE SET count=count+1",
                    (domain,)
                )
            row = con.execute("SELECT count FROM flags WHERE domain = ?", (domain,)).fetchone()
            flag_count = row["count"] if row else 0

        con.execute(
            "INSERT INTO history (ts, snippet, verdict, credibility, domain, confidence, claim, reasoning) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)",
            (text[:120], verdict_display, credibility_pct, domain or None, v_conf, text[:120], v_reasoning[:200])
        )
        con.commit()
    finally:
        con.close()

    related = [
        {"title": a["title"], "link": a.get("link", a.get("url", "")), "source": a.get("source_domain", a.get("publisher", ""))}
        for a in (source_evidence or [])[:5]
    ] if verdict_display != "Likely real" else []

    return jsonify(
        verdict=v_verdict,
        verdict_display=verdict_display,
        verdict_badge=v_verdict,
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
        source_signal=round(source_signal, 3),
        claim=v_claim,
        input_type=input_type,
        category=category_info,
        confidence=v_conf,
        reasoning=v_reasoning,
        evidence=v_evidence,
        guardrail_notes=v_guardrails,
        style_indicator=style_indicator,
        source_domain=domain or None,
        source_reputation=source_reputation,
        cached=v_cached,
        checked_at=v_checked_at
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
    if not ADMIN_API_KEY:
        return jsonify(error="Service Unavailable: Retraining endpoint is disabled because ADMIN_API_KEY is not configured."), 503

    auth_key = request.headers.get("X-Admin-Key") or (request.get_json(force=True, silent=True) or {}).get("admin_key")
    if auth_key != ADMIN_API_KEY:
        return jsonify(error="Unauthorized: Administrative authentication required."), 401

    if not ENABLE_LEARNING:
        status_info = learn.status()
        status_info["error"] = "Forbidden: Model retraining and harvesting is disabled by configuration."
        return jsonify(started=False, **status_info), 403

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
