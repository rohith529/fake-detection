"""Unit tests for verify_llm module with mocked Gemini client."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
import verify_llm
import db


@pytest.fixture(autouse=True)
def init_clean_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_history.db"
    monkeypatch.setattr(db, "DB_FILE", test_db)
    db.init_db()
    yield


def test_missing_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = verify_llm.verify_claim("NASA launched a spacecraft to Europa in 2024")
    assert res["verdict"] == "UNAVAILABLE"
    assert res["confidence"] == "low"
    assert "not configured" in res["reasoning"]
    assert res["cached"] is False


def test_gemini_timeout_error_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-api-key")
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = TimeoutError("Connection timed out after 25s")

        res = verify_llm.verify_claim("A major earthquake occurred near Japan yesterday")
        assert res["verdict"] == "UNAVAILABLE"
        assert res["confidence"] == "low"
        assert "temporarily unavailable" in res["reasoning"].lower()
        # Verify 2 attempts were made (initial + 1 retry)
        assert mock_client.models.generate_content.call_count == 2


def test_successful_gemini_verification_and_grounding_filter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-api-key")

    mock_llm_json = json.dumps({
        "claim": "India launched Chandrayaan-3 successfully in 2023",
        "verdict": "SUPPORTED",
        "confidence": "high",
        "reasoning": "Multiple official news outlets verified the successful landing.",
        "evidence": [
            {
                "url": "https://www.thehindu.com/sci-tech/chandrayaan-3-landing",
                "publisher": "The Hindu",
                "title": "Chandrayaan-3 touches down on lunar south pole",
                "published_date": "2023-08-23",
                "stance": "supports",
                "key_sentence": "ISRO landed the rover on the lunar surface."
            },
            {
                "url": "https://fake-hallucinated-domain.org/moon-landing-fake",
                "publisher": "Hallucinated Source",
                "title": "Fake Moon Article",
                "published_date": "2023-08-23",
                "stance": "supports",
                "key_sentence": "Hallucinated text."
            },
            {
                "url": "https://bbc.com/news/world-asia-india-chandrayaan-3",
                "publisher": "BBC",
                "title": "Chandrayaan-3: India makes historic moon landing",
                "published_date": "2023-08-23",
                "stance": "supports",
                "key_sentence": "India becomes the first nation to land near moon south pole."
            }
        ]
    })

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock response candidates with grounding metadata containing only the real URLs
        mock_resp = MagicMock()
        mock_resp.text = f"```json\n{mock_llm_json}\n```"
        
        mock_candidate = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.web.uri = "https://www.thehindu.com/sci-tech/chandrayaan-3-landing"
        mock_chunk2 = MagicMock()
        mock_chunk2.web.uri = "https://bbc.com/news/world-asia-india-chandrayaan-3"
        
        mock_candidate.grounding_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]
        mock_resp.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_resp

        res = verify_llm.verify_claim("India launched Chandrayaan-3 successfully in 2023", today="2026-11-21")
        
        # Check verdict and confidence
        assert res["verdict"] == "SUPPORTED"
        assert res["confidence"] == "high"
        
        # Check that the hallucinated URL was dropped
        evidence_urls = [e["url"] for e in res["evidence"]]
        assert "https://www.thehindu.com/sci-tech/chandrayaan-3-landing" in evidence_urls
        assert "https://bbc.com/news/world-asia-india-chandrayaan-3" in evidence_urls
        assert "https://fake-hallucinated-domain.org/moon-landing-fake" not in evidence_urls
        assert len(res["evidence"]) == 2


def test_verification_caching_returns_cached_hit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-api-key")

    mock_llm_json = json.dumps({
        "claim": "Water boils at 100 degrees Celsius at standard atmospheric pressure",
        "verdict": "SUPPORTED",
        "confidence": "high",
        "reasoning": "Scientific consensus across textbooks and encyclopedias.",
        "evidence": [
            {
                "url": "https://britannica.com/science/water",
                "publisher": "Encyclopaedia Britannica",
                "title": "Water boiling point",
                "published_date": "2020-01-01",
                "stance": "supports",
                "key_sentence": "Water boils at 100 C."
            },
            {
                "url": "https://wikipedia.org/wiki/Water",
                "publisher": "Wikipedia",
                "title": "Water",
                "published_date": "2020-01-01",
                "stance": "supports",
                "key_sentence": "Boiling point 100 C."
            }
        ]
    })

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.text = mock_llm_json
        mock_resp.candidates = []
        mock_client.models.generate_content.return_value = mock_resp

        claim_text = "Water boils at 100 degrees Celsius at standard atmospheric pressure"
        # First call: cache miss -> calls Gemini
        res1 = verify_llm.verify_claim(claim_text, today="2026-11-21")
        assert res1["cached"] is False
        assert mock_client.models.generate_content.call_count == 1

        # Second call: identical -> cache hit -> does NOT call Gemini
        res2 = verify_llm.verify_claim(claim_text, today="2026-11-21")
        assert res2["cached"] is True
        assert res2["verdict"] == res1["verdict"]
        assert mock_client.models.generate_content.call_count == 1
