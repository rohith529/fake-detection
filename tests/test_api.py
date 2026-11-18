"""Integration tests for Flask API endpoints."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
from app import app, ADMIN_API_KEY


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_analyze_endpoint_success(client):
    payload = {
        "text": "The Reserve Bank of India announced that repo rates will remain steady at 6.5 percent",
        "url": "https://www.thehindu.com"
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200
    d = r.get_json()
    assert "verdict" in d
    assert "credibility" in d
    assert "p_real" in d
    assert "explanation" in d
    assert "source_evidence" in d
    assert 0 <= d["credibility"] <= 100


def test_analyze_endpoint_too_short(client):
    r = client.post("/api/analyze", json={"text": "Too short"})
    assert r.status_code == 400
    assert "at least 5 words" in r.get_json()["error"]


def test_analyze_endpoint_too_long(client):
    r = client.post("/api/analyze", json={"text": "word " * 1200})
    assert r.status_code == 400
    assert "exceeds maximum length" in r.get_json()["error"]


def test_history_endpoint(client):
    r = client.get("/api/history")
    assert r.status_code == 200
    assert isinstance(r.get_json(), list)


def test_feedback_staging(client):
    # Staged feedback should succeed and not poison training immediately
    r = client.post("/api/feedback", json={
        "text": "This is a verifiable news claim that was misclassified",
        "label": 1
    })
    assert r.status_code == 200
    d = r.get_json()
    assert d.get("saved") is True
    assert "pending_feedback" in d


def test_retrain_unauthorized(client):
    r = client.post("/api/retrain", json={"harvest": False})
    assert r.status_code == 401
    assert "Unauthorized" in r.get_json()["error"]


def test_analyze_uncorroborated_claim_capped_at_unverified(client):
    # A claim with formal vocabulary but no corroboration from trusted sources must be capped at 50%
    payload = {
        "text": "former cm of telangana KCR passed away today evening",
        "url": ""
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "Unverified"
    assert d["credibility"] <= 50
    assert d["uncorroborated_flag"] is True


def test_retrain_authorized_header(client):
    # Authorized with admin key
    r = client.post(
        "/api/retrain",
        headers={"X-Admin-Key": ADMIN_API_KEY},
        json={"harvest": False}
    )
    # Either 200 (started) or 409 (already running)
    assert r.status_code in (200, 409)
