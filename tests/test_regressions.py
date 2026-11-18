"""Regression tests for known historic failure modes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import app
from preprocess import clean


def test_fact_check_prefix_not_penalized_in_clean():
    headline = "Fact Check: NASA did not discover alien spacecraft on the moon"
    cleaned = clean(headline)
    # Ensure no synthetic bias tokens were injected
    assert "__DLEAD__" not in cleaned
    assert "__QLEAD__" not in cleaned
    # Ensure negation survived
    assert "not" in cleaned.split()


def test_negation_inversion_tokens():
    claim_affirmative = "Vaccines cause chronic illness"
    claim_negative = "Vaccines do not cause chronic illness"

    tokens_aff = set(clean(claim_affirmative).split())
    tokens_neg = set(clean(claim_negative).split())

    # The negative version MUST contain negation and differ from affirmative
    assert "not" in tokens_neg
    assert "not" not in tokens_aff
    assert tokens_aff != tokens_neg


def test_high_concurrency_sqlite_queries():
    import concurrent.futures
    import db

    def query_history():
        con = db.get_db()
        try:
            rows = con.execute("SELECT count(*) as c FROM history").fetchone()
            return rows["c"]
        finally:
            con.close()

    # Run 20 concurrent readers to verify no "database is locked" errors
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(query_history) for _ in range(20)]
        results = [f.result() for f in futures]
    assert len(results) == 20


def test_entity_divergence_prevents_false_corroboration():
    from similarity import classify_stance
    claim = "former cm of telangana KCR passed away"
    evidence_headline = "Telangana: Former CM KCR's sister passes away in Hyderabad"
    stance, conf, agree = classify_stance(claim, evidence_headline, topical_similarity=0.65)
    # Even with high word overlap, relative mismatch must classify as NEUTRAL
    assert stance == "NEUTRAL"
    assert agree == 0.50
