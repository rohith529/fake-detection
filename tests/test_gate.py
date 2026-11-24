"""Unit tests for the input classification gate (backend/gate.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
from gate import classify_input, classify_category


class TestInputGate:
    # ── 1. Too Short (< 5 words) ─────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "Rain today",
        "Breaking news alert",
        "Just four small words",
    ])
    def test_too_short(self, text):
        res = classify_input(text)
        assert res["type"] == "too_short"
        assert len(res["reason"]) > 0

    # ── 2. Questions ─────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "Will it rain today in Mumbai?",
        "Will it rain today in Mumbai",  # starts with 'will', no '?'
        "Is the stock market open today in New York?",
        "Are interest rates going to increase?",
        "Does the Ministry of Finance approve this?",
        "Did the president sign the executive order?",
        "Can anyone verify this breaking report?",
        "Should investors be concerned about inflation?",
        "How did the accident on the highway happen?",
        "What caused the global power outage yesterday?",
        "When was the election commission formed?",
        "Where was the international summit conducted?",
        "Who is the current prime minister of the country?",
        "Why did the space agency postpone the launch?",
        '"Will it rain today in London?"',  # fully quoted question
        "“Did the governor resign this morning?”",
    ])
    def test_questions(self, text):
        res = classify_input(text)
        assert res["type"] == "question"
        assert len(res["reason"]) > 0

    # ── 3. Predictions ───────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "Heavy rains will hit coastal regions over the weekend",
        "The economy is going to enter recession according to economists",
        "Global temperatures are expected to reach record highs",
        "Weather forecast predicts heavy thunderstorms across the region",
        "Gold prices are projected to rise significantly by year end",
        "The committee will announce the verdict tomorrow morning",
        "Flights will resume normal operations next week",
    ])
    def test_predictions(self, text):
        res = classify_input(text)
        assert res["type"] == "prediction"
        assert len(res["reason"]) > 0

    # ── 4. Opinions ──────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "Editorial: The new budget policy fails the working class entirely",
        "Opinion: Why space exploration funding should be doubled immediately",
        "Op-ed: Higher interest rates will damage small business recovery",
        "I think the city administration made a grave error yesterday",
        "In my view, the newly introduced policy is fundamentally flawed",
        "In my opinion, this remains the worst healthcare bill in decades",
        "We believe the regulatory body overstepped its legal jurisdiction",
        "The proposed changes are, in my view, detrimental to public health",
        "The council members, I strongly believe, ignored critical community feedback",
    ])
    def test_opinions(self, text):
        res = classify_input(text)
        assert res["type"] == "opinion"
        assert len(res["reason"]) > 0

    # ── 5. Verifiable Factual Claims (Allowed) ──────────────────────────────
    @pytest.mark.parametrize("text", [
        "The Reserve Bank of India kept repo rates steady at 6.5 percent",
        "NASA successfully landed the Perseverance rover on Mars in 2021",
        "Scientists confirmed that hot water does not cure all cancers",
        "The Ministry of Finance issued a circular regarding foreign remittances",
        "The supreme court delivered a landmark verdict on privacy rights",
    ])
    def test_claims(self, text):
        res = classify_input(text)
        assert res["type"] == "claim"

    # ── 6. Edge Cases ────────────────────────────────────────────────────────
    def test_edge_case_will_as_legal_noun_in_past_context(self):
        # "will" refers to last will and testament in past tense
        text = "The court confirmed that the deceased industrialist signed his last will and testament in 2015"
        res = classify_input(text)
        assert res["type"] == "claim"

    def test_edge_case_will_as_noun_phrase(self):
        # "demonstrated strong political will"
        text = "The prime minister demonstrated strong political will during the historic treaty negotiations"
        res = classify_input(text)
        assert res["type"] == "claim"

    def test_edge_case_proper_name_will_smith(self):
        # "Will" is part of proper name "Will Smith"
        text = "Will Smith won the Academy Award for Best Actor in 2022"
        res = classify_input(text)
        assert res["type"] == "claim"

    def test_edge_case_quoted_question_inside_factual_claim(self):
        # Quoted question embedded inside a declarative reporting claim
        text = 'The defense spokesperson asked "Will any nation tolerate terrorism?" at the UN summit yesterday.'
        res = classify_input(text)
        assert res["type"] == "claim"

    def test_edge_case_past_progressive_was_going_to(self):
        # "was going to" in a completed narrative context is past, not a future prediction
        text = "The minister was going to the airport when the motorcade was stopped yesterday"
        res = classify_input(text)
        assert res["type"] == "claim"

    # ── 7. Broader Daily Activity Categories ─────────────────────────────────
    @pytest.mark.parametrize("text,expected_cat", [
        ("hyderabad noticed heavy rainfall on 24th november 2026", "weather"),
        ("Severe cyclonic storm forecast in coastal Andhra Pradesh", "weather"),
        ("India defeated Australia in the cricket world cup final yesterday", "sports"),
        ("Lionel Messi scored two goals to win the championship match", "sports"),
        ("Shah Rukh Khan announced his new movie trailer release date", "entertainment"),
        ("Hollywood actor won the best actor award at the film festival", "entertainment"),
        ("Supreme Court delivers historic verdict on electoral bonds scheme", "politics"),
        ("Prime Minister inaugurated the new parliament building session", "politics"),
        ("Scientists confirm discovery of new water ice deposits on the Moon", "health_science"),
        ("Reserve Bank of India maintained repo rate at 6.5 percent", "business"),
        ("Major earthquake hits northern region measuring 6.4 on the Richter scale", "crime_disaster"),
    ])
    def test_classify_categories(self, text, expected_cat):
        cat_info = classify_category(text)
        assert cat_info["id"] == expected_cat
        assert len(cat_info["label"]) > 0
        assert len(cat_info["icon"]) > 0
