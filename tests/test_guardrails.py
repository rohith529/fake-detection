"""Unit tests for deterministic verification guardrails (backend/verify_llm.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
from verify_llm import apply_guardrails


class TestGuardrails:
    # ── Rule 1: Empty Evidence Downgrade ─────────────────────────────────────
    def test_empty_evidence_downgrades_supported(self):
        result = {
            "claim": "The Reserve Bank announced rate cuts",
            "verdict": "SUPPORTED",
            "confidence": "high",
            "reasoning": "Believed to be true.",
            "evidence": []
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="The Reserve Bank announced rate cuts")
        assert guarded["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert guarded["confidence"] == "low"
        assert "Downgraded: No verifiable evidence sources found." in guarded["reasoning"]
        assert any("No verifiable evidence" in note for note in guarded["guardrail_notes"])

    def test_empty_evidence_downgrades_refuted(self):
        result = {
            "claim": "Drinking bleach cures illness",
            "verdict": "REFUTED",
            "confidence": "high",
            "reasoning": "Believed to be false.",
            "evidence": []
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="Drinking bleach cures illness")
        assert guarded["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert guarded["confidence"] == "low"
        assert "Downgraded: No verifiable evidence sources found." in guarded["reasoning"]

    # ── Rule 2: Time-Sensitive Outdated Evidence ─────────────────────────────
    def test_time_sensitive_claim_outdated_evidence_downgraded(self):
        # Claim contains "today", evidence is 5 days old
        result = {
            "claim": "Heavy rain hit Hyderabad today",
            "verdict": "SUPPORTED",
            "confidence": "high",
            "reasoning": "Found news articles.",
            "evidence": [
                {
                    "url": "https://thehindu.com/news/1",
                    "publisher": "The Hindu",
                    "title": "Rains lash Hyderabad",
                    "published_date": "2026-11-15",  # 6 days before 2026-11-21
                    "stance": "supports",
                    "key_sentence": "Rains lashed parts of the city."
                }
            ]
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="Heavy rain hit Hyderabad today")
        assert guarded["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert guarded["confidence"] == "low"
        assert any("older than 2 days" in note for note in guarded["guardrail_notes"])

    def test_time_sensitive_claim_unknown_dates_caps_confidence(self):
        result = {
            "claim": "Heavy rain is falling in Hyderabad now",
            "verdict": "SUPPORTED",
            "confidence": "high",
            "reasoning": "Articles found.",
            "evidence": [
                {
                    "url": "https://thehindu.com/news/1",
                    "publisher": "The Hindu",
                    "title": "Rains in Hyderabad",
                    "published_date": "unknown",
                    "stance": "supports",
                    "key_sentence": "Rains reported."
                }
            ]
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="Heavy rain is falling in Hyderabad now")
        assert guarded["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert guarded["confidence"] == "low"
        assert any("unknown publication dates" in note for note in guarded["guardrail_notes"])

    def test_time_sensitive_claim_fresh_evidence_kept(self):
        result = {
            "claim": "Heavy rain hit Hyderabad today",
            "verdict": "SUPPORTED",
            "confidence": "medium",
            "reasoning": "Articles found.",
            "evidence": [
                {
                    "url": "https://thehindu.com/news/1",
                    "publisher": "The Hindu",
                    "title": "Rains in Hyderabad today",
                    "published_date": "2026-11-21",  # Fresh!
                    "stance": "supports",
                    "key_sentence": "Rains lashed city today."
                }
            ]
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="Heavy rain hit Hyderabad today")
        assert guarded["verdict"] == "SUPPORTED"
        assert guarded["confidence"] == "medium"

    # ── Rule 3: Stance Mismatch ──────────────────────────────────────────────
    def test_supported_verdict_without_supports_stance_downgraded(self):
        result = {
            "claim": "Tech company announced layoffs",
            "verdict": "SUPPORTED",
            "confidence": "medium",
            "reasoning": "Article mentions tech company.",
            "evidence": [
                {
                    "url": "https://reuters.com/tech",
                    "publisher": "Reuters",
                    "title": "Tech company annual report",
                    "published_date": "2026-11-20",
                    "stance": "neutral",  # NEUTRAL, not supports!
                    "key_sentence": "Company published revenue numbers."
                }
            ]
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="Tech company announced layoffs")
        assert guarded["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert any("no evidence items had stance 'supports'" in note for note in guarded["guardrail_notes"])

    def test_refuted_verdict_without_refutes_stance_downgraded(self):
        result = {
            "claim": "Government eliminated all income tax",
            "verdict": "REFUTED",
            "confidence": "medium",
            "reasoning": "Tax laws continue.",
            "evidence": [
                {
                    "url": "https://bbc.com/business",
                    "publisher": "BBC",
                    "title": "Global tax updates",
                    "published_date": "2026-11-20",
                    "stance": "neutral",
                    "key_sentence": "Discussion on tax systems."
                }
            ]
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="Government eliminated all income tax")
        assert guarded["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert any("no evidence items had stance 'refutes'" in note for note in guarded["guardrail_notes"])

    # ── Rule 4: Single Publisher Domain High Confidence Lowered ──────────────
    def test_high_confidence_with_single_domain_lowered_to_medium(self):
        result = {
            "claim": "New archaeological discovery in Greece",
            "verdict": "SUPPORTED",
            "confidence": "high",
            "reasoning": "Two articles found.",
            "evidence": [
                {
                    "url": "https://reuters.com/article-1",
                    "publisher": "Reuters",
                    "title": "Ancient tomb unearthed",
                    "published_date": "2026-11-20",
                    "stance": "supports",
                    "key_sentence": "Tomb was discovered."
                },
                {
                    "url": "https://reuters.com/article-2",
                    "publisher": "Reuters",
                    "title": "Greek ministry comments on tomb",
                    "published_date": "2026-11-20",
                    "stance": "supports",
                    "key_sentence": "Ministry verified findings."
                }
            ]
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="New archaeological discovery in Greece")
        assert guarded["confidence"] == "medium"
        assert any("fewer than 2 distinct publisher domains" in note for note in guarded["guardrail_notes"])

    def test_high_confidence_with_multiple_domains_preserved(self):
        result = {
            "claim": "New archaeological discovery in Greece",
            "verdict": "SUPPORTED",
            "confidence": "high",
            "reasoning": "Two independent articles found.",
            "evidence": [
                {
                    "url": "https://reuters.com/article-1",
                    "publisher": "Reuters",
                    "title": "Ancient tomb unearthed",
                    "published_date": "2026-11-20",
                    "stance": "supports",
                    "key_sentence": "Tomb was discovered."
                },
                {
                    "url": "https://bbc.com/news/world-1",
                    "publisher": "BBC",
                    "title": "Greek ministry comments on tomb",
                    "published_date": "2026-11-20",
                    "stance": "supports",
                    "key_sentence": "Ministry verified findings."
                }
            ]
        }
        guarded = apply_guardrails(result, today="2026-11-21", claim_text="New archaeological discovery in Greece")
        assert guarded["confidence"] == "high"

    # ── Rule 5: User URL Domain (Satire / Disreputable) ─────────────────────
    def test_satire_domain_forces_not_a_claim(self):
        result = {
            "claim": "Congress Passes Bill Legalizing Naptime",
            "verdict": "SUPPORTED",
            "confidence": "high",
            "reasoning": "Article reported on website.",
            "evidence": [
                {
                    "url": "https://theonion.com/naptime-bill",
                    "publisher": "The Onion",
                    "title": "Congress Passes Bill",
                    "published_date": "2026-11-20",
                    "stance": "supports",
                    "key_sentence": "Bill was signed into law."
                }
            ]
        }
        guarded = apply_guardrails(
            result,
            today="2026-11-21",
            claim_text="Congress Passes Bill Legalizing Naptime",
            user_url="https://www.theonion.com/naptime-bill"
        )
        assert guarded["verdict"] == "NOT_A_CLAIM"
        assert guarded["confidence"] == "high"
        assert "satire" in guarded["reasoning"].lower()
        assert any("satire publication" in note for note in guarded["guardrail_notes"])

    def test_disreputable_domain_forces_not_enough_evidence(self):
        result = {
            "claim": "Secret chemicals detected in water supply",
            "verdict": "SUPPORTED",
            "confidence": "high",
            "reasoning": "Article reported on site.",
            "evidence": []
        }
        guarded = apply_guardrails(
            result,
            today="2026-11-21",
            claim_text="Secret chemicals detected in water supply",
            user_url="https://infowars.com/article-water"
        )
        assert guarded["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert guarded["confidence"] == "low"
        assert any("disreputable source" in note for note in guarded["guardrail_notes"])
