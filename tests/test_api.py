"""Integration tests for Flask API endpoints."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
from app import app, ADMIN_API_KEY


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_analyze_endpoint_success_schema(client):
    payload = {
        "text": "The Reserve Bank of India kept repo rates steady at 6.5 percent",
        "url": "https://www.thehindu.com"
    }
    
    mock_llm_result = {
        "claim": "RBI announced repo rates remain steady at 6.5 percent",
        "verdict": "SUPPORTED",
        "confidence": "high",
        "reasoning": "Official RBI announcement verified by multiple national outlets.",
        "evidence": [
            {
                "url": "https://www.thehindu.com/business/rbi-policy",
                "publisher": "The Hindu",
                "title": "RBI monetary policy update",
                "published_date": "2024-02-08",
                "stance": "supports",
                "key_sentence": "Repo rate kept at 6.5 percent."
            },
            {
                "url": "https://reuters.com/markets/rbi-rates",
                "publisher": "Reuters",
                "title": "India central bank maintains rate",
                "published_date": "2024-02-08",
                "stance": "supports",
                "key_sentence": "Rates held unchanged."
            }
        ],
        "guardrail_notes": [],
        "cached": False,
        "checked_at": "2026-11-21T12:00:00Z"
    }

    with patch("verify_llm.verify_claim", return_value=mock_llm_result):
        r = client.post("/api/analyze", json=payload)
        assert r.status_code == 200
        d = r.get_json()
        
        # Verify schema requirements
        assert d["claim"] == mock_llm_result["claim"]
        assert d["input_type"] == "claim"
        assert d["verdict"] in ("SUPPORTED", "REFUTED", "NOT_ENOUGH_EVIDENCE", "NOT_A_CLAIM", "UNAVAILABLE")
        assert d["confidence"] in ("low", "medium", "high")
        assert isinstance(d["reasoning"], str)
        assert isinstance(d["evidence"], list)
        assert isinstance(d["guardrail_notes"], list)
        assert "style_indicator" in d
        assert d["style_indicator"]["label"] in ("neutral", "sensational")
        assert 0.0 <= d["style_indicator"]["score"] <= 1.0
        assert "note" in d["style_indicator"]
        assert d["source_domain"] == "thehindu.com"
        assert d["source_reputation"] == "trusted"
        assert isinstance(d["cached"], bool)
        assert "checked_at" in d
        assert "latency_ms" in d


def test_analyze_endpoint_gate_question_bypasses_llm(client):
    payload = {
        "text": "Will it rain today in Hyderabad?",
        "url": ""
    }
    with patch("verify_llm.verify_claim") as mock_verify:
        r = client.post("/api/analyze", json=payload)
        assert r.status_code == 200
        d = r.get_json()
        assert d["input_type"] == "question"
        assert d["verdict"] == "NOT_A_CLAIM"
        assert "question" in d["reasoning"].lower()
        assert mock_verify.call_count == 0  # LLM must NOT be called for questions
        assert d["latency_ms"] < 200


def test_analyze_endpoint_empty_input(client):
    r = client.post("/api/analyze", json={"text": ""})
    assert r.status_code == 400
    assert "provide a headline" in r.get_json()["error"].lower()


def test_analyze_endpoint_too_long(client):
    r = client.post("/api/analyze", json={"text": "word " * 1200})
    assert r.status_code == 400
    assert "exceeds maximum length" in r.get_json()["error"]


def test_history_endpoint(client):
    r = client.get("/api/history")
    assert r.status_code == 200
    assert isinstance(r.get_json(), list)


def test_feedback_staging(client):
    r = client.post("/api/feedback", json={
        "text": "This is a verifiable news claim that was misclassified",
        "label": 1
    })
    assert r.status_code == 200
    d = r.get_json()
    assert d.get("saved") is True
    assert "pending_feedback" in d


def test_retrain_disabled_when_no_admin_key(client, monkeypatch):
    monkeypatch.setattr("app.ADMIN_API_KEY", None)
    r = client.post("/api/retrain", json={"harvest": False})
    assert r.status_code == 503
    assert "Service Unavailable" in r.get_json()["error"]


def test_retrain_unauthorized(client, monkeypatch):
    monkeypatch.setattr("app.ADMIN_API_KEY", "test-secret-key")
    r = client.post("/api/retrain", json={"harvest": False})
    assert r.status_code == 401
    assert "Unauthorized" in r.get_json()["error"]


def test_retrain_disabled_by_feature_flag(client, monkeypatch):
    monkeypatch.setattr("app.ADMIN_API_KEY", "test-secret-key")
    monkeypatch.setattr("app.ENABLE_LEARNING", False)
    r = client.post(
        "/api/retrain",
        headers={"X-Admin-Key": "test-secret-key"},
        json={"harvest": False}
    )
    assert r.status_code == 403
    assert "Forbidden" in r.get_json()["error"]


def test_retrain_authorized_when_learning_enabled(client, monkeypatch):
    monkeypatch.setattr("app.ADMIN_API_KEY", "test-secret-key")
    monkeypatch.setattr("app.ENABLE_LEARNING", True)
    r = client.post(
        "/api/retrain",
        headers={"X-Admin-Key": "test-secret-key"},
        json={"harvest": False}
    )
    assert r.status_code in (200, 409)
