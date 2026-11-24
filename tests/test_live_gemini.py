"""Live end-to-end integration test with Google Gemini Search Grounding.
Skipped automatically when GEMINI_API_KEY is not set.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
import verify_llm


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or "your_" in os.getenv("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY") == "dummy-api-key",
    reason="Valid GEMINI_API_KEY not set; skipping live Gemini API test."
)
def test_live_gemini_search_grounding():
    # Well-established scientific fact
    text = "The speed of light in a vacuum is approximately 299,792 kilometers per second"
    res = verify_llm.verify_claim(text)
    
    assert res["verdict"] in ("SUPPORTED", "NOT_ENOUGH_EVIDENCE")
    assert res["confidence"] in ("medium", "high")
    assert isinstance(res["reasoning"], str)
    assert len(res["reasoning"]) > 10
    if res["evidence"]:
        for ev in res["evidence"]:
            assert ev["url"].startswith("http")
            assert ev["stance"] in ("supports", "refutes", "neutral")
