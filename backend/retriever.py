"""Retriever: fetch live articles from trusted sources via Google News RSS.

Remediated architecture:
1. Robust Domain Resolution & Zero Fallback:
   Google News RSS frequently obscures entry.source.href. This module parses canonical
   publisher identities from XML source tags, title suffixes, and link structures,
   mapping them directly to verified domains in sources.json.
2. Intelligent Query Formulation:
   Replaces naive 10-word slicing with salience-based keyword extraction.
   Removes boilerplate, filters stopwords, and isolates entities to generate
   focused 3-to-5 keyword search queries.
3. Persistent On-Disk Caching:
   Uses SQLite-backed cache (with 24h TTL) to survive server restarts and prevent
   rate-limiting by upstream providers.
"""
import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import db

SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text())
_log = logging.getLogger(__name__)

# Map publisher names (as present in Google News RSS title/source tags) to canonical domains
PUBLISHER_NAME_MAP = {
    "the hindu": "thehindu.com",
    "hindu": "thehindu.com",
    "the indian express": "indianexpress.com",
    "indian express": "indianexpress.com",
    "hindustan times": "hindustantimes.com",
    "ht": "hindustantimes.com",
    "ndtv": "ndtv.com",
    "ndtv news": "ndtv.com",
    "the times of india": "timesofindia.indiatimes.com",
    "times of india": "timesofindia.indiatimes.com",
    "toi": "timesofindia.indiatimes.com",
    "reuters": "reuters.com",
    "thomson reuters": "reuters.com",
    "associated press": "apnews.com",
    "ap news": "apnews.com",
    "ap": "apnews.com",
    "bbc": "bbc.com",
    "bbc news": "bbc.com",
    "bbc hindi": "bbc.com",
    "pib": "pib.gov.in",
    "press information bureau": "pib.gov.in",
    "the onion": "theonion.com",
    "infowars": "infowars.com"
}

_BOILERPLATE_RE = re.compile(
    r"^(fact\s*check[:\s]|viral\s*test[:\s]|is\s+it\s+true\s+that|did\s+|breaking[:\s]|exclusive[:\s])",
    re.I
)


def _domain_of(url: str) -> str:
    """Extract bare domain from URL without 'www.' prefix."""
    if not url:
        return ""
    host = urlparse(url if "//" in url else "//" + url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def resolve_canonical_domain(entry) -> str:
    """Resolve trusted domain through source href, source text, title suffix, or entry link."""
    # 1. Check entry.source.url / href
    source_obj = getattr(entry, "source", None) or entry.get("source", {})
    if isinstance(source_obj, dict):
        href = source_obj.get("url") or source_obj.get("href") or ""
        title = (source_obj.get("title") or source_obj.get("value") or "").strip().lower()
    else:
        href = getattr(source_obj, "href", "") or getattr(source_obj, "url", "")
        title = getattr(source_obj, "title", "") or getattr(source_obj, "value", "") or ""
        title = str(title).strip().lower()

    domain = _domain_of(href)
    for known in SOURCES:
        if domain == known or domain.endswith("." + known):
            return known

    # 2. Check source name in publisher map
    if title and title in PUBLISHER_NAME_MAP:
        return PUBLISHER_NAME_MAP[title]

    # 3. Check trailing publisher in headline: "Headline Text - Publisher"
    full_title = getattr(entry, "title", "")
    if " - " in full_title:
        pub_suffix = full_title.rsplit(" - ", 1)[-1].strip().lower()
        if pub_suffix in PUBLISHER_NAME_MAP:
            return PUBLISHER_NAME_MAP[pub_suffix]
        # Also check substring match
        for name, dom in PUBLISHER_NAME_MAP.items():
            if name in pub_suffix:
                return dom

    # 4. Fallback to entry.link if direct
    link_domain = _domain_of(getattr(entry, "link", ""))
    for known in SOURCES:
        if link_domain == known or link_domain.endswith("." + known):
            return known

    return ""


def formulate_search_query(text: str) -> str:
    """Extract focused 3-to-5 keywords for Google News RSS retrieval."""
    # Strip common debunking/question prefixes that distract search
    clean_text = _BOILERPLATE_RE.sub("", text.strip())
    
    # Extract candidate words
    words = re.findall(r"[A-Za-z0-9]+", clean_text)
    
    # Separate named entities / capitalized terms from lower words
    salient = []
    seen = set()
    
    # Priority 1: Capitalized or alphanumeric identifiers (Proper nouns, agencies)
    for w in words:
        if (w[0].isupper() or any(c.isdigit() for c in w)) and w.lower() not in ENGLISH_STOP_WORDS:
            low = w.lower()
            if low not in seen and len(low) > 2:
                seen.add(low)
                salient.append(w)
                if len(salient) >= 4:
                    break

    # Priority 2: Informative content words if query is still too short
    if len(salient) < 4:
        for w in words:
            low = w.lower()
            if low not in seen and low not in ENGLISH_STOP_WORDS and len(low) > 3:
                seen.add(low)
                salient.append(w)
                if len(salient) >= 5:
                    break

    query = " ".join(salient) if salient else " ".join(words[:4])
    return query


def fetch(claim_text: str) -> list[dict]:
    """Return up to 8 trusted articles for the claim.
    
    Queries Google News RSS with persistent caching and canonical domain extraction.
    Each item: {title: str, link: str, source_domain: str, reliability: float}
    """
    query = formulate_search_query(claim_text)
    if not query.strip():
        return []

    cache_key = "rss:" + re.sub(r"\s+", "_", query.lower().strip())
    
    # 1. Check persistent SQLite cache
    cached = db.cache_get(cache_key)
    if cached is not None:
        return cached

    # 2. Perform external RSS query
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            timeout=5,
            params=dict(q=query, hl="en-IN", gl="IN", ceid="IN:en"),
            headers={"User-Agent": "VeritasDesk/2.0 (+https://github.com/academic/fake-news-detector)"}
        )
        feed = feedparser.parse(r.content)

        results = []
        for e in feed.entries:
            title = e.title.rsplit(" - ", 1)[0].strip()
            link = e.link
            domain = resolve_canonical_domain(e)
            reliability = SOURCES.get(domain)

            if reliability is None or reliability < 0.70:
                continue

            results.append({
                "title": title,
                "link": link,
                "source_domain": domain,
                "reliability": reliability,
            })

        final_articles = results[:8]
        # Store in persistent cache with 24-hour TTL
        db.cache_set(cache_key, final_articles, ttl_seconds=86400)
        return final_articles

    except Exception:
        _log.exception("retriever: fetch failed")
        return []
