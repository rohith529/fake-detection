"""Gemini LLM Verification Module with Google Search Grounding and Guardrails.

Provides:
- verify_claim(text, url=None, today=None) -> dict
- apply_guardrails(result, today, claim_text, user_url=None) -> dict (pure function)
- PROMPT_VERSION constant
"""
import copy
from datetime import datetime, date, timezone
import hashlib
import json
import logging
import os
import re
import urllib.parse
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    import db
except ImportError:
    from backend import db

load_dotenv()

_log = logging.getLogger("veritas.verify")
PROMPT_VERSION = "2026-11-21.v1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Known satire and disreputable sources
SATIRE_DOMAINS = {
    "theonion.com",
    "babylonbee.com",
    "clickhole.com",
    "theborowitzreport.com",
    "newsthump.com",
    "fakingnews.com"
}

DISREPUTABLE_DOMAINS = {
    "infowars.com",
    "naturalnews.com",
    "worldnewsdailyreport.com",
    "beforeitsnews.com"
}


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    url: str = Field(description="Actual URL found via search")
    publisher: str = Field(description="Name of publisher/source")
    title: str = Field(description="Headline of the source article")
    published_date: str = Field(default="unknown", description="ISO date YYYY-MM-DD or 'unknown'")
    stance: Literal["supports", "refutes", "neutral"] = Field(description="Stance toward the claim")
    key_sentence: str = Field(description="Relevant quotation or tight paraphrase from source")


class VerificationResult(BaseModel):
    claim: str = Field(description="The specific factual claim extracted from the input")
    verdict: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_EVIDENCE", "NOT_A_CLAIM"] = Field(
        description="Verification verdict"
    )
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence level")
    reasoning: str = Field(description="Concise explanation based ONLY on evidence found")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="List of evidence items")


# ── Helper Functions ─────────────────────────────────────────────────────────

def extract_domain(url_str: str) -> str:
    """Extract clean domain without www from a URL."""
    if not url_str:
        return ""
    try:
        host = urllib.parse.urlparse(url_str if "//" in url_str else "//" + url_str).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def normalize_url(u: str) -> str:
    """Normalize a URL for robust cross-checking against grounding chunks."""
    u = (u or "").strip().lower()
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix):]
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


def parse_iso_date(d_str: str) -> Optional[date]:
    """Parse an ISO date string into a date object, returning None if invalid or unknown."""
    if not d_str or str(d_str).strip().lower() in ("unknown", "null", "none", ""):
        return None
    d_clean = str(d_str).split("T")[0].strip()
    try:
        return datetime.strptime(d_clean, "%Y-%m-%d").date()
    except Exception:
        return None


def clean_json_response(text: str) -> str:
    """Strip code fences from LLM response text if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── Pure Deterministic Guardrails ────────────────────────────────────────────

def apply_guardrails(
    result: dict,
    today: str,
    claim_text: str,
    user_url: Optional[str] = None
) -> dict:
    """Pure deterministic guardrail function that inspects and adjusts verification output.
    
    Rules:
    1. Empty evidence with SUPPORTED/REFUTED -> downgrade to NOT_ENOUGH_EVIDENCE, low conf.
    2. Time-sensitive claims: evidence older than 2 days (or 7 days) -> downgrade; all unknown dates -> cap low.
    3. Stance mismatch: SUPPORTED without 'supports' or REFUTED without 'refutes' -> downgrade.
    4. Confidence 'high' with < 2 distinct publisher domains -> lower to 'medium'.
    5. Satire/disreputable user domain -> force NOT_A_CLAIM (satire) or NOT_ENOUGH_EVIDENCE (disreputable).
    """
    res = copy.deepcopy(result)
    notes = list(res.get("guardrail_notes", []))
    
    evidence = res.get("evidence", [])
    verdict = res.get("verdict", "NOT_ENOUGH_EVIDENCE")
    confidence = res.get("confidence", "low")
    reasoning = res.get("reasoning", "")

    # Rule 5: User-provided URL domain check (satire / disreputable)
    if user_url:
        domain = extract_domain(user_url)
        is_satire = any(domain == s or domain.endswith("." + s) for s in SATIRE_DOMAINS)
        is_disreputable = any(domain == d or domain.endswith("." + d) for d in DISREPUTABLE_DOMAINS)
        
        if is_satire:
            res["verdict"] = "NOT_A_CLAIM"
            res["confidence"] = "high"
            res["reasoning"] = f"Domain '{domain}' is a known satire/parody publication. Satirical content is not a factual claim."
            notes.append(f"Source domain '{domain}' is identified as a satire publication; forced verdict to NOT_A_CLAIM.")
            res["guardrail_notes"] = notes
            return res
        elif is_disreputable:
            res["verdict"] = "NOT_ENOUGH_EVIDENCE"
            res["confidence"] = "low"
            res["reasoning"] = f"Domain '{domain}' is flagged as an unreliable or disreputable source. Cannot verify claim based on this source."
            notes.append(f"Source domain '{domain}' is a flagged disreputable source; forced verdict to NOT_ENOUGH_EVIDENCE.")
            res["guardrail_notes"] = notes
            return res

    # Rule 1: Empty evidence with SUPPORTED or REFUTED
    if not evidence and verdict in ("SUPPORTED", "REFUTED"):
        res["verdict"] = "NOT_ENOUGH_EVIDENCE"
        res["confidence"] = "low"
        res["reasoning"] = (reasoning + " Downgraded: No verifiable evidence sources found.").strip()
        notes.append("Downgraded: No verifiable evidence sources found.")
        verdict = "NOT_ENOUGH_EVIDENCE"

    # Rule 2: Time sensitivity check
    today_dt = parse_iso_date(today) or datetime.now(timezone.utc).date()
    claim_lower = (claim_text or "").lower()
    
    short_time_markers = (
        "today", "now", "currently", "this morning", "tonight", "this afternoon",
        "yesterday", "last night", "earlier today", "this evening"
    )
    week_time_markers = ("this week", "recent", "recently")
    
    date_patterns = [
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?\b",
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b",
        r"\b202[4-9]\b"
    ]
    has_calendar_date = any(re.search(pat, claim_lower) for pat in date_patterns)

    is_short_time = any(m in claim_lower for m in short_time_markers) or has_calendar_date
    is_week_time = any(m in claim_lower for m in week_time_markers)
    
    if is_short_time or is_week_time:
        max_days = 2 if is_short_time else 7
        parsed_dates = [parse_iso_date(e.get("published_date")) for e in evidence]
        
        all_unknown = all(d is None for d in parsed_dates) if evidence else True
        if all_unknown and evidence:
            if res.get("confidence") in ("high", "medium"):
                res["confidence"] = "low"
                notes.append("All evidence sources have unknown publication dates for a time-sensitive claim; confidence capped at low.")

        all_outdated = True
        for d in parsed_dates:
            if d is not None:
                age_days = (today_dt - d).days
                if age_days <= max_days and age_days >= 0:
                    all_outdated = False
                    break
        
        if all_outdated and verdict == "SUPPORTED" and evidence:
            res["verdict"] = "NOT_ENOUGH_EVIDENCE"
            res["confidence"] = "low"
            downgrade_msg = f"Downgraded: Claim is time-sensitive but evidence sources are older than {max_days} days or dates are unknown."
            res["reasoning"] = (reasoning + f" {downgrade_msg}").strip()
            notes.append(f"Claim is time-sensitive, but all evidence dates are older than {max_days} days or unknown.")
            verdict = "NOT_ENOUGH_EVIDENCE"

    # Rule 3: Stance mismatch
    if verdict == "SUPPORTED":
        has_support = any(str(e.get("stance", "")).lower() == "supports" for e in evidence)
        if not has_support:
            res["verdict"] = "NOT_ENOUGH_EVIDENCE"
            res["confidence"] = "low"
            res["reasoning"] = (res.get("reasoning", "") + " Downgraded: No evidence sources have stance 'supports'.").strip()
            notes.append("Downgraded: Verdict was SUPPORTED but no evidence items had stance 'supports'.")
            verdict = "NOT_ENOUGH_EVIDENCE"
    elif verdict == "REFUTED":
        has_refute = any(str(e.get("stance", "")).lower() == "refutes" for e in evidence)
        if not has_refute:
            res["verdict"] = "NOT_ENOUGH_EVIDENCE"
            res["confidence"] = "low"
            res["reasoning"] = (res.get("reasoning", "") + " Downgraded: No evidence sources have stance 'refutes'.").strip()
            notes.append("Downgraded: Verdict was REFUTED but no evidence items had stance 'refutes'.")
            verdict = "NOT_ENOUGH_EVIDENCE"

    # Rule 4: Publisher diversity check for 'high' confidence
    if res.get("confidence") == "high":
        domains = {extract_domain(e.get("url", "")) for e in evidence}
        domains.discard("")
        if len(domains) < 2:
            res["confidence"] = "medium"
            notes.append("Lowered confidence from high to medium because fewer than 2 distinct publisher domains were cited.")

    res["guardrail_notes"] = notes
    return res


# ── System Instruction Builder ───────────────────────────────────────────────

def build_system_instruction(today_str: str, user_url: Optional[str] = None) -> str:
    """Build the versioned fact-checker prompt with injected clock date and instructions."""
    url_context = ""
    if user_url:
        url_context = (
            f"\nUSER CONTEXT URL: {user_url}\n"
            "Note: The source domain must not by itself be treated as proof of truth. "
            "Satire sites (e.g. theonion.com, babylonbee.com) are NEVER factual sources and must be evaluated as NOT_A_CLAIM.\n"
        )

    return f"""Today's date is strictly {today_str}. (Version: {PROMPT_VERSION})
You are Veritas Desk, a rigorous, neutral, professional fact-checking engine.
Use Google Search grounding to discover real-time, authoritative evidence. Do NOT answer from memory alone.

Evaluation Rules:
1. Identify the specific factual claim made in the user text.
2. Determine whether the text contains a verifiable factual statement.
   - Non-claims (questions, forecasts, opinions, rhetorical satire) MUST be given verdict NOT_A_CLAIM.
3. Allowed verdicts:
   - SUPPORTED: Verified true by authoritative, corroborating reporting.
   - REFUTED: Debunked or directly contradicted by credible sources.
   - NOT_ENOUGH_EVIDENCE: Insufficient reporting, ambiguous sources, conflicting reports, or unconfirmed rumors. "NOT_ENOUGH_EVIDENCE" is explicitly preferred over guessing.
   - NOT_A_CLAIM: Questions, personal opinions, forecasts, or satire.
4. Topical similarity is NOT support: An article that covers the same event, person, or weather location without confirming or debunking the specific claim must NOT be counted as supporting evidence.
5. Time sensitivity: If the claim refers to "today", "now", "this week", or recent breaking events, sources older than 2 days (for "today/now") or 7 days (for "this week") do NOT count as evidence for the current state.
6. Confidence:
   - "high": Corroborated by multiple independent, highly reputable news organizations.
   - "medium": Single credible source with clear reporting.
   - "low": Limited evidence, ambiguous phrasing, or conflicting details.
{url_context}
Output Format:
You must reply ONLY with a valid JSON object matching this schema:
{{
  "claim": "<specific factual claim extracted from input>",
  "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_EVIDENCE" | "NOT_A_CLAIM",
  "confidence": "low" | "medium" | "high",
  "reasoning": "<concise explanation based strictly on evidence found>",
  "evidence": [
    {{
      "url": "<actual URL found via search>",
      "publisher": "<publisher name>",
      "title": "<headline of the source article>",
      "published_date": "<ISO date YYYY-MM-DD or 'unknown'>",
      "stance": "supports" | "refutes" | "neutral",
      "key_sentence": "<verbatim or tight paraphrase of the relevant sentence from the source>"
    }}
  ]
}}"""


# ── Core Verification Pipeline ───────────────────────────────────────────────

def verify_claim(
    text: str,
    url: Optional[str] = None,
    today: Optional[str] = None
) -> dict:
    """Verify a factual claim using Gemini with Google Search grounding and deterministic guardrails.
    
    Results are cached in SQLite verify_cache (6h TTL for time-sensitive, 24h otherwise).
    If GEMINI_API_KEY is not set or calls fail, returns verdict UNAVAILABLE.
    """
    cleaned_text = (text or "").strip()
    user_domain = extract_domain(url) if url else ""
    today_str = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Check persistent SQLite cache
    cache_key = hashlib.sha256(
        f"{cleaned_text.lower()}|{user_domain.lower()}|{PROMPT_VERSION}|{today_str}".encode()
    ).hexdigest()

    cached_row = db.verify_cache_get(cache_key)
    if cached_row:
        res = copy.deepcopy(cached_row)
        res["cached"] = True
        if "checked_at" not in res:
            res["checked_at"] = datetime.now(timezone.utc).isoformat()
        return res

    # 2. Check URL domain first (Rule 5 Guardrail: Satire / Disreputable check)
    if url:
        domain = extract_domain(url)
        is_satire = any(domain == s or domain.endswith("." + s) for s in SATIRE_DOMAINS)
        is_disreputable = any(domain == d or domain.endswith("." + d) for d in DISREPUTABLE_DOMAINS)
        if is_satire:
            return {
                "claim": cleaned_text,
                "verdict": "NOT_A_CLAIM",
                "confidence": "high",
                "reasoning": f"Domain '{domain}' is a known satire/parody publication. Satirical content is not a factual claim.",
                "evidence": [],
                "guardrail_notes": [f"Source domain '{domain}' is identified as a satire publication; forced verdict to NOT_A_CLAIM."],
                "cached": False,
                "checked_at": datetime.now(timezone.utc).isoformat()
            }
        elif is_disreputable:
            return {
                "claim": cleaned_text,
                "verdict": "NOT_ENOUGH_EVIDENCE",
                "confidence": "low",
                "reasoning": f"Domain '{domain}' is flagged as an unreliable or disreputable source. Cannot verify claim based on this source.",
                "evidence": [],
                "guardrail_notes": [f"Source domain '{domain}' is a flagged disreputable source; forced verdict to NOT_ENOUGH_EVIDENCE."],
                "cached": False,
                "checked_at": datetime.now(timezone.utc).isoformat()
            }

    # 3. Check API key
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or "your_" in api_key.lower() or "placeholder" in api_key.lower():
        return {
            "claim": cleaned_text,
            "verdict": "UNAVAILABLE",
            "confidence": "low",
            "reasoning": "Gemini API key is not configured (or is a placeholder) in GEMINI_API_KEY. Live verification is unavailable.",
            "evidence": [],
            "guardrail_notes": ["Verification service unavailable: missing or placeholder GEMINI_API_KEY."],
            "cached": False,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

    # 3. Call Gemini with Google Search grounding and 1 retry
    system_instruction = build_system_instruction(today_str, url)
    user_prompt = f"Verify this input:\n\n{cleaned_text}"
    if url:
        user_prompt += f"\nSource URL: {url}"

    parsed_result = None
    grounding_uris = set()

    for attempt in range(2):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            
            # Configure Google Search grounding with JSON output
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=25.0)
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=config
            )

            # Extract grounding metadata URLs
            if response.candidates:
                cand = response.candidates[0]
                meta = getattr(cand, "grounding_metadata", None)
                if meta and getattr(meta, "grounding_chunks", None):
                    for chunk in meta.grounding_chunks:
                        web = getattr(chunk, "web", None)
                        if web and getattr(web, "uri", None):
                            grounding_uris.add(web.uri)

            raw_text = response.text or ""
            clean_json = clean_json_response(raw_text)
            parsed_data = json.loads(clean_json)
            
            # Validate with Pydantic
            validated = VerificationResult(**parsed_data)
            parsed_result = validated.model_dump()
            break
        except Exception as e:
            _log.warning("Gemini verification attempt %d failed: %s", attempt + 1, e)
            if attempt == 1:
                return {
                    "claim": cleaned_text,
                    "verdict": "UNAVAILABLE",
                    "confidence": "low",
                    "reasoning": f"Verification service temporarily unavailable: {type(e).__name__} - {str(e)}",
                    "evidence": [],
                    "guardrail_notes": ["Verification service error or timeout after retry."],
                    "cached": False,
                    "checked_at": datetime.now(timezone.utc).isoformat()
                }

    if not parsed_result:
        return {
            "claim": cleaned_text,
            "verdict": "UNAVAILABLE",
            "confidence": "low",
            "reasoning": "Verification failed to produce a structured response.",
            "evidence": [],
            "guardrail_notes": ["Empty or invalid response from verification provider."],
            "cached": False,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

    # 4. Cross-check evidence URLs against grounding metadata
    # Drop hallucinated URLs not discovered during Google search
    filtered_evidence = []
    normalized_grounding = {normalize_url(u) for u in grounding_uris}
    
    for item in parsed_result.get("evidence", []):
        item_norm = normalize_url(item.get("url", ""))
        # If grounding URIs were discovered, ensure the URL was in them
        if normalized_grounding:
            matched = any(
                item_norm == gu or (item_norm and item_norm in gu) or (gu and gu in item_norm)
                for gu in normalized_grounding
            )
            if matched:
                filtered_evidence.append(item)
            else:
                _log.info("Dropped hallucinated URL not in grounding metadata: %s", item.get("url"))
        else:
            # If no grounding chunks returned by API, keep evidence item
            filtered_evidence.append(item)

    parsed_result["evidence"] = filtered_evidence

    # 5. Apply deterministic guardrails
    final_result = apply_guardrails(parsed_result, today_str, cleaned_text, user_url=url)

    # 6. Cache result with appropriate TTL
    time_sensitive_markers = ("today", "now", "currently", "this morning", "tonight", "this afternoon", "yesterday", "tomorrow")
    is_time_sensitive = any(m in cleaned_text.lower() for m in time_sensitive_markers)
    ttl_seconds = 6 * 3600 if is_time_sensitive else 24 * 3600

    db.verify_cache_set(
        cache_key=cache_key,
        result_dict=final_result,
        ttl_seconds=ttl_seconds,
        claim_text=cleaned_text,
        url_domain=user_domain,
        prompt_version=PROMPT_VERSION,
        date_str=today_str
    )

    final_result["cached"] = False
    final_result["checked_at"] = datetime.now(timezone.utc).isoformat()
    return final_result
